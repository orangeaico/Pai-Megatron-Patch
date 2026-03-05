"""
Loss mask processors for fine-tuning data preparation.

This module provides a modular system for determining which token indices
should have their loss mask set to 0 during training.
"""

from typing import List, Dict, Any, Protocol, Optional, Set
from abc import ABC, abstractmethod
import ast
import tokenize
import io

# Global debug flag
DEBUG_MODE = False

# Unicode character corruption mappings
# Maps corrupted characters to their correct emoji equivalents
# 
# To add new emoji mappings:
# 1. Find the corrupted characters in debug output (e.g., 'âľħ')
# 2. Identify the correct emoji it should be (e.g., '✅') 
# 3. Add mapping: 'corrupted_chars': 'correct_emoji'
#
# Common sources of corruption:
# - Emoji characters getting mangled during tokenization/decoding
# - UTF-8 encoding issues with special characters
# - Multi-byte Unicode characters being split incorrectly
EMOJI_CORRUPTION_MAP = {
    'âľħ': '✅',  # WHITE HEAVY CHECK MARK (U+2705)
    'âĿĮ': '❌',  # CROSS MARK (U+274C)
    'âľĵ': '✓',  # CHECK MARK (U+2713)
    'âľĹ': '✗',  # BALLOT X (U+2717)
    'âĶĶâĶĢâĶĢ': '└──',  # BOX DRAWINGS LIGHT UP AND RIGHT + HORIZONTAL (tree branch)
    'âĨĴ': '→',  # RIGHTWARDS ARROW (U+2192)
    'ĉ': '\t',  # TAB CHARACTER (U+0009)
    'âĶĤ': '│',  # BOX DRAWINGS LIGHT VERTICAL (U+2502)
    # Add more mappings here as they are discovered
    # Example:
    # 'âł': '❗',  # Heavy exclamation mark
    # 'âĸ': '⚠️',   # Warning sign
}

# Keywords used to identify source file paths in str_replace actions.
# DiffProcessor only processes str_replace actions on files whose paths
# contain at least one of these keywords. Add new keywords here to expand coverage.
SOURCE_FILE_KEYWORDS = [
    'xarray',
]


def apply_unicode_corrections(text: str) -> str:
    """Apply Unicode corruption corrections to text.
    
    Args:
        text: Text that may contain corrupted Unicode characters
        
    Returns:
        Text with corrupted characters replaced with correct emojis
    """
    corrected_text = text
    for corrupted, correct in EMOJI_CORRUPTION_MAP.items():
        corrected_text = corrected_text.replace(corrupted, correct)
    return corrected_text

def set_debug_mode(enabled: bool):
    """Enable or disable debug logging."""
    global DEBUG_MODE
    DEBUG_MODE = enabled


class LossMaskProcessor(ABC):
    """Abstract base class for loss mask processors."""
    
    def __init__(self):
        self.mask_value = -1  # Default mask value
    
    @abstractmethod
    def process(self, tokenizer, token_ids: List[int], msg_dict: Dict[str, Any], 
                assistant_turn: int) -> List[int]:
        """
        Process the message and return loss mask array.
        
        Args:
            tokenizer: The tokenizer instance
            token_ids: List of token IDs for the message
            msg_dict: Dictionary containing 'role' and 'content' fields
            assistant_turn: The turn number for assistant messages (1-indexed)
            
        Returns:
            List of same length as token_ids where:
            - mask_value at indices that should be masked
            - -1 at all other indices
        """
        pass


class StrReplaceEditorProcessor(LossMaskProcessor):
    """
    Processor that checks for str_replace_editor actions and extracts fields.
    
    Parses XML content to determine command type and extract relevant fields.
    - For create commands: extracts file_text and identifies comment lines
    - For str_replace commands: extracts old_str and new_str
    - For view commands: returns empty list (ignore)
    """
    
    def __init__(self):
        super().__init__()
        self.mask_value = 0  # Set mask value to 0 for this processor
    
    def _extract_parameter(self, content: str, param_name: str) -> Optional[str]:
        """Extract parameter value from XML content."""
        param_tag = f'<parameter={param_name}>'
        if param_tag not in content:
            return None
            
        param_start = content.find(param_tag) + len(param_tag)
        param_end = content.find('</parameter>', param_start)
        
        if param_end == -1:
            return None
            
        return content[param_start:param_end]
    
    def _strip_newlines(self, text: str) -> str:
        """Strip leading and trailing newlines from text."""
        if text.startswith('\n'):
            text = text[1:]
        if text.endswith('\n'):
            text = text[:-1]
        return text
    
    def _check_command_type(self, content: str, command: str) -> bool:
        """Check if content contains a command parameter with optional newlines."""
        # Check all possible variations with newlines
        variations = [
            f'<parameter=command>{command}</parameter>',
            f'<parameter=command>\n{command}</parameter>',
            f'<parameter=command>{command}\n</parameter>',
            f'<parameter=command>\n{command}\n</parameter>'
        ]
        return any(var in content for var in variations)
    
    def _get_comment_lines_from_ast(self, code: str) -> Set[int]:
        """Use AST to identify lines with comments (docstrings)."""
        comment_lines = set()
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                # Check for docstrings in functions, classes, and modules
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    # Get the first statement in the body
                    if node.body and isinstance(node.body[0], ast.Expr):
                        value = node.body[0].value
                        if isinstance(value, (ast.Str, ast.Constant)) and isinstance(value.value if hasattr(value, 'value') else value.s, str):
                            # This is a docstring
                            comment_lines.add(node.body[0].lineno)
                            # Multi-line docstrings may span several lines
                            if hasattr(node.body[0], 'end_lineno') and node.body[0].end_lineno:
                                for line_num in range(node.body[0].lineno, node.body[0].end_lineno + 1):
                                    comment_lines.add(line_num)
                elif isinstance(node, ast.Module):
                    # Module-level docstring
                    if node.body and isinstance(node.body[0], ast.Expr):
                        value = node.body[0].value
                        if isinstance(value, (ast.Str, ast.Constant)) and isinstance(value.value if hasattr(value, 'value') else value.s, str):
                            comment_lines.add(node.body[0].lineno)
                            if hasattr(node.body[0], 'end_lineno') and node.body[0].end_lineno:
                                for line_num in range(node.body[0].lineno, node.body[0].end_lineno + 1):
                                    comment_lines.add(line_num)
        except:
            # If AST parsing fails, return empty set
            pass
        
        return comment_lines
    
    def _get_comment_lines_from_tokenizer(self, code: str) -> Set[int]:
        """Use tokenizer to identify lines with # comments (only pure comment lines, not inline comments)."""
        comment_lines = set()
        
        try:
            lines = code.split('\n')
            for line_num, line in enumerate(lines, 1):
                # Strip whitespace to check content
                stripped = line.lstrip()
                # Check if line starts with # (after stripping whitespace)
                if stripped.startswith('#'):
                    comment_lines.add(line_num)
        except:
            # If processing fails, return what we have
            pass
            
        return comment_lines
    
    def _get_all_comment_lines(self, code: str) -> Set[int]:
        """Get all comment lines including both # comments and docstrings."""
        # Get # comments using tokenizer
        hash_comments = self._get_comment_lines_from_tokenizer(code)
        # Get docstrings using AST
        docstrings = self._get_comment_lines_from_ast(code)
        # Combine both
        return hash_comments.union(docstrings)
    
    def _find_parameter_boundaries(self, tokenizer, token_ids: List[int], param_name: str, 
                                     window_size: int = 6) -> Optional[tuple[int, int]]:
        """
        Find the start and end positions of a parameter in the token sequence.
        
        Args:
            tokenizer: The tokenizer instance
            token_ids: List of token IDs to search in
            param_name: The parameter name to search for (e.g., 'file_text')
            window_size: Number of consecutive tokens to join for searching
            
        Returns:
            A tuple of (start_idx, end_idx) where:
            - start_idx is the position after <parameter=param_name>
            - end_idx is the position of </parameter>
            Returns None if parameter not found
        """
        # Convert all tokens to their string representation with special chars
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        
        # Clean tokens - convert special characters to actual representation
        cleaned_tokens = []
        for token in tokens:
            # Handle common tokenizer special characters
            clean = token.replace('Ġ', ' ')  # GPT-style space
            clean = clean.replace('▁', ' ')  # SentencePiece-style space
            clean = clean.replace('Ċ', '\n')  # Newline
            cleaned_tokens.append(clean)
        
        # Search for opening tag
        open_pattern = f'<parameter={param_name}>'
        start_idx = None
        
        # Find start position
        for i in range(len(cleaned_tokens) - window_size + 1):
            # Join consecutive tokens
            joined_text = ''.join(cleaned_tokens[i:i + window_size])
            
            # Check if search pattern is in the joined text
            if open_pattern in joined_text:
                # Find the exact position within the window
                prefix_len = joined_text.find(open_pattern) + len(open_pattern)
                
                # Estimate which token contains the end of the pattern
                cumulative_len = 0
                for j in range(window_size):
                    cumulative_len += len(cleaned_tokens[i + j])
                    if cumulative_len >= prefix_len:
                        # Found the position just after the parameter tag
                        start_idx = i + j + 1
                        break
                
                if start_idx is None:
                    # Fallback: use middle of window
                    start_idx = i + window_size // 2
                break
        
        if start_idx is None:
            return None
        
        # Search for closing tag
        close_pattern = '</parameter>'
        end_idx = None
        
        # Search from start_idx onwards
        for i in range(start_idx + 5, len(cleaned_tokens) - 4):
            # Join a few tokens to check for the pattern
            joined_text = ''.join(cleaned_tokens[i:i+4])
            if close_pattern in joined_text:
                # Found the closing tag
                end_idx = i
                break
        
        if end_idx is None:
            # If no closing tag found, return None (invalid parameter)
            return None
            
        return (start_idx, end_idx)
    
    def _find_token_indices_for_lines(self, tokenizer, token_ids: List[int], content: str, 
                                     target_lines: Set[int], file_text: str, offset: int = 0) -> List[int]:
        """Find token indices that correspond to specific lines in the file_text.
        
        Processes all comments (docstrings and single-line) in chronological order.
        
        Args:
            tokenizer: The tokenizer
            token_ids: List of token IDs to search within
            content: Not used in this version
            target_lines: Set of line numbers that contain comments
            file_text: The actual file text
            offset: Offset to add to returned indices (for when token_ids is a subset)
        
        Returns:
            List of token indices that should be masked
        """
        if not target_lines:
            return []
        
        file_lines = file_text.split('\n')
        masked_indices = []
        processed_lines = set()
        search_start_pos = 0
        
        # Process all comment lines in order
        sorted_lines = sorted(target_lines)
        
        # Group consecutive docstring lines together
        comment_groups = []
        i = 0
        while i < len(sorted_lines):
            line_num = sorted_lines[i]
            if line_num > len(file_lines):
                i += 1
                continue
                
            line_content = file_lines[line_num - 1]
            
            # Check if this is the start of a docstring
            if '"""' in line_content or "'''" in line_content:
                # Find the quote type
                quote_type = '"""' if '"""' in line_content else "'''"
                
                # Check if the docstring starts and ends on the same line
                quote_count = line_content.count(quote_type)
                if quote_count >= 2:
                    # Single-line docstring
                    comment_groups.append(('docstring', [line_num]))
                    i += 1
                else:
                    # Multi-line docstring - collect consecutive lines until closing quote
                    docstring_lines = [line_num]
                    j = i + 1
                    found_closing = False
                    
                    # Look for consecutive lines in target_lines that continue the docstring
                    while j < len(sorted_lines):
                        next_line_num = sorted_lines[j]
                        
                        # Check if this is the next consecutive line
                        expected_line = docstring_lines[-1] + 1
                        if next_line_num != expected_line:
                            # Not consecutive - this docstring is incomplete, treat first line as single comment
                            break
                        
                        if next_line_num <= len(file_lines):
                            next_content = file_lines[next_line_num - 1]
                            docstring_lines.append(next_line_num)
                            
                            # Check if this line closes the docstring
                            if quote_type in next_content:
                                found_closing = True
                                j += 1
                                break
                        j += 1
                    
                    if found_closing and len(docstring_lines) > 1:
                        comment_groups.append(('docstring', docstring_lines))
                        i = j
                    else:
                        # Treat as single line comment if no proper closing found
                        comment_groups.append(('single', [line_num]))
                        i += 1
            else:
                # Single line comment
                comment_groups.append(('single', [line_num]))
                i += 1
        
        if DEBUG_MODE:
            print(f"Processing {len(comment_groups)} comment groups in chronological order")
        
        # Process all comments in order
        for comment_type, lines in comment_groups:
            if comment_type == 'docstring':
                # Process docstring block
                first_line = lines[0] - 1  # 0-indexed
                last_line = lines[-1] - 1   # 0-indexed
                
                # Get the exact docstring text
                docstring_lines_text = file_lines[first_line:last_line + 1]
                docstring_text = '\n'.join(docstring_lines_text)
                
                if DEBUG_MODE:
                    print(f"Processing docstring block with {len(lines)} lines: {lines[:3]}...")
                
                # Tokenize the entire docstring
                docstring_tokens = tokenizer.encode(docstring_text, add_special_tokens=False)
                
                if not docstring_tokens:
                    continue
                
                # Try to find this token sequence
                found = False
                for start_offset in range(min(3, len(docstring_tokens))):
                    if found:
                        break
                    for end_offset in range(min(3, len(docstring_tokens))):
                        if found:
                            break
                        
                        search_tokens = docstring_tokens[start_offset:len(docstring_tokens)-end_offset if end_offset > 0 else None]
                        
                        if not search_tokens or len(search_tokens) < 3:
                            continue
                        
                        # Look for this sequence in token_ids
                        for i in range(search_start_pos, len(token_ids) - len(search_tokens) + 1):
                            if token_ids[i:i+len(search_tokens)] == search_tokens:
                                # Found match
                                found = True
                                start = i - start_offset
                                end = i + len(search_tokens) + end_offset
                                
                                # Ensure we don't go out of bounds
                                start = max(0, start)
                                end = min(len(token_ids), end)
                                
                                # Extend for whitespace as before
                                tokens = tokenizer.convert_ids_to_tokens(token_ids)
                                
                                # Extend backwards
                                for j in range(start - 1, max(0, start - 10), -1):
                                    if j < len(tokens):
                                        token_text = tokens[j]
                                        clean = token_text.replace('Ġ', ' ').replace('Ċ', '\n').replace('▁', ' ')
                                        if '\n' in clean:
                                            break
                                        if clean.strip() == '':
                                            start = j
                                        else:
                                            break
                                
                                # Check if we need to extend forward
                                if end > 0 and end <= len(tokens):
                                    last_token = tokens[end - 1]
                                    last_clean = last_token.replace('Ġ', ' ').replace('Ċ', '\n').replace('▁', ' ')
                                    if '\n' not in last_clean:
                                        # Extend forwards for newline
                                        for j in range(end, min(len(tokens), end + 5)):
                                            if j < len(tokens):
                                                token_text = tokens[j]
                                                clean = token_text.replace('Ġ', ' ').replace('Ċ', '\n').replace('▁', ' ')
                                                if '\n' in clean and clean.strip() == '':
                                                    end = j + 1
                                                    break
                                                elif clean.strip() == '':
                                                    end = j + 1
                                                else:
                                                    break
                                
                                # Add tokens
                                for idx in range(start, end):
                                    masked_indices.append(idx + offset)
                                
                                search_start_pos = end
                                break
                
                if not found:
                    print(f"WARNING: Could not find docstring in token sequence! Lines: {lines}")
                    return []
            
            else:
                # Process single line comment
                line_num = lines[0]
                comment_line = file_lines[line_num - 1]
                
                if not comment_line and not comment_line.strip():
                    continue
                
                # Tokenize the comment line
                comment_tokens = tokenizer.encode(comment_line, add_special_tokens=False)
                
                if not comment_tokens:
                    continue
                
                if DEBUG_MODE and line_num == 1:
                    print(f"DEBUG: Processing line {line_num}: {repr(comment_line)}")
                    print(f"  Tokenized to {len(comment_tokens)} tokens: {comment_tokens}")
                    print(f"  Search starting at position {search_start_pos} in {len(token_ids)} total tokens")
                
                # Try to find this token sequence
                found = False
                for start_offset in range(min(3, len(comment_tokens))):
                    if found:
                        break
                    for end_offset in range(min(3, len(comment_tokens))):
                        if found:
                            break
                        
                        search_tokens = comment_tokens[start_offset:len(comment_tokens)-end_offset if end_offset > 0 else None]
                        
                        if not search_tokens:
                            continue
                        
                        # Look for this sequence in token_ids
                        for i in range(search_start_pos, len(token_ids) - len(search_tokens) + 1):
                            if token_ids[i:i+len(search_tokens)] == search_tokens:
                                # Found match
                                found = True
                                start = i - start_offset
                                end = i + len(search_tokens) + end_offset
                                
                                # Extend for whitespace as before
                                tokens = tokenizer.convert_ids_to_tokens(token_ids)
                                
                                # Extend backwards
                                for j in range(start - 1, max(0, start - 10), -1):
                                    if j < len(tokens):
                                        token_text = tokens[j]
                                        clean = token_text.replace('Ġ', ' ').replace('Ċ', '\n').replace('▁', ' ')
                                        if '\n' in clean:
                                            break
                                        if clean.strip() == '':
                                            start = j
                                        else:
                                            break
                                
                                # Check if we need to extend forward
                                if end > 0 and end <= len(tokens):
                                    last_token = tokens[end - 1]
                                    last_clean = last_token.replace('Ġ', ' ').replace('Ċ', '\n').replace('▁', ' ')
                                    if '\n' not in last_clean:
                                        # Extend forwards for newline
                                        for j in range(end, min(len(tokens), end + 5)):
                                            if j < len(tokens):
                                                token_text = tokens[j]
                                                clean = token_text.replace('Ġ', ' ').replace('Ċ', '\n').replace('▁', ' ')
                                                if '\n' in clean and clean.strip() == '':
                                                    end = j + 1
                                                    break
                                                elif clean.strip() == '':
                                                    end = j + 1
                                                else:
                                                    break
                                
                                # Add tokens
                                for idx in range(start, end):
                                    masked_indices.append(idx + offset)
                                
                                search_start_pos = end
                                break
                
                if not found:
                    print(f"WARNING: Could not find line {line_num} in token sequence: {repr(comment_line)}")
                    return []
        
        # Remove duplicates and sort
        masked_indices = sorted(set(masked_indices))
        return masked_indices
    
    def process(self, tokenizer, token_ids: List[int], msg_dict: Dict[str, Any], 
                assistant_turn: int) -> List[int]:
        """
        Parse str_replace_editor actions and extract fields.
        
        Args:
            tokenizer: The tokenizer instance
            token_ids: List of token IDs for the message
            msg_dict: Dictionary containing 'role' and 'content' fields
            assistant_turn: The turn number for assistant messages
            
        Returns:
            Empty list if str_replace_editor view command is found,
            None to continue processing for other cases
        """
        content = msg_dict.get('content', '')
        
        # Check if content contains str_replace_editor function call
        if '<function=str_replace_editor>' not in content:
            return None
            
        # Determine the command type
        is_create = False
        is_str_replace = False
        is_view = False
        
        # Check for command parameter with variations
        if self._check_command_type(content, 'create'):
            is_create = True
        elif self._check_command_type(content, 'str_replace'):
            is_str_replace = True
        elif self._check_command_type(content, 'view'):
            is_view = True
        else:
            # Fallback: check if it's str_replace by looking for old_str parameter
            if '<parameter=old_str>' in content:
                is_str_replace = True
            # Check if it's create by looking for file_text parameter
            elif '<parameter=file_text>' in content:
                is_create = True
        
        # Handle view command - return empty list to ignore
        if is_view:
            return []
        
        # Handle create command
        if is_create:
            file_text = self._extract_parameter(content, 'file_text')
            if file_text is not None:
                # Strip newlines for processing
                file_text_stripped = self._strip_newlines(file_text)
                if DEBUG_MODE:
                    print(f"StrReplaceEditorProcessor: Found create command (assistant turn {assistant_turn})")
                
                # Get all comment lines
                comment_lines = self._get_all_comment_lines(file_text_stripped)
                
                
                if comment_lines:
                    if DEBUG_MODE:
                        print(f"Found {len(comment_lines)} comment lines")
                        # Print the actual comment lines
                        file_lines = file_text_stripped.split('\n')
                        print(f"Comment lines being masked:")
                        for line_num in sorted(comment_lines):
                            if 1 <= line_num <= len(file_lines):
                                comment_content = file_lines[line_num - 1]  # Don't strip - keep whitespace
                                # Only print if line has content (even if it's just whitespace)
                                if comment_content or comment_content.strip():
                                    print(f"  Line {line_num}: {repr(comment_content)}")
                    
                    # First, we need to find where file_text appears in the token sequence
                    # Use the new robust method to find parameter boundaries
                    boundaries = self._find_parameter_boundaries(tokenizer, token_ids, 'file_text')
                    
                    if boundaries is None:
                        # Always print warnings, regardless of debug mode
                        print(f"WARNING: Could not find create command file_text content in token sequence")
                        return []
                    
                    file_text_start_idx, file_text_end_idx = boundaries
                    
                    if DEBUG_MODE:
                        print(f"Found file_text boundaries: start={file_text_start_idx}, end={file_text_end_idx}")
                    
                    # Extract only the tokens that are part of file_text
                    file_text_token_ids = token_ids[file_text_start_idx:file_text_end_idx]
                    
                    # Find token indices corresponding to comment lines within file_text
                    masked_indices = self._find_token_indices_for_lines(
                        tokenizer, file_text_token_ids, content, comment_lines, file_text_stripped, 
                        offset=file_text_start_idx
                    )
                    
                    # Always calculate expected character count for validation
                    total_comment_chars = 0
                    file_lines = file_text_stripped.split('\n')
                    for line_num in sorted(comment_lines):
                        if 1 <= line_num <= len(file_lines):
                            # Include the line content plus newline
                            total_comment_chars += len(file_lines[line_num - 1]) + 1  # +1 for newline
                    
                    if masked_indices:
                        
                        # Calculate total characters in masked tokens
                        tokens = tokenizer.convert_ids_to_tokens(token_ids)
                        total_masked_chars = 0
                        for idx in masked_indices:
                            if 0 <= idx < len(tokens):
                                token_text = tokens[idx]
                                clean = token_text.replace('Ġ', ' ').replace('Ċ', '\n').replace('▁', ' ')
                                total_masked_chars += len(clean)
                        
                        if DEBUG_MODE:
                            print(f"Masking {len(masked_indices)} tokens for comment lines")
                            # Print the actual tokens being masked
                            print(f"Tokens being masked:")
                            
                            # Group consecutive tokens
                            if masked_indices:
                                consecutive_groups = []
                                current_group = [masked_indices[0]]
                                
                                for i in range(1, len(masked_indices)):
                                    if masked_indices[i] == masked_indices[i-1] + 1:
                                        # Consecutive token
                                        current_group.append(masked_indices[i])
                                    else:
                                        # New group
                                        consecutive_groups.append(current_group)
                                        current_group = [masked_indices[i]]
                                
                                # Add the last group
                                consecutive_groups.append(current_group)
                                
                                # Print each group
                                for group in consecutive_groups:
                                    # Build the text for this group
                                    group_text = ""
                                    for idx in group:
                                        if 0 <= idx < len(tokens):
                                            token_text = tokens[idx]
                                            # Convert to readable format
                                            clean = token_text.replace('Ġ', ' ').replace('Ċ', '\n').replace('▁', ' ')
                                            group_text += clean
                                    
                                    # Print with token range
                                    if len(group) == 1:
                                        print(f"  Token {group[0]}: {repr(group_text)}")
                                    else:
                                        print(f"  Tokens {group[0]}-{group[-1]}: {repr(group_text)}")
                                
                                # Show character count comparison in debug mode
                                print(f"\nCharacter count comparison:")
                                print(f"  Total characters in identified comments: {total_comment_chars}")
                                print(f"  Total characters in masked tokens: {total_masked_chars}")
                                
                                if total_comment_chars > 0:
                                    diff_percent = abs(total_masked_chars - total_comment_chars) / total_comment_chars * 100
                                    threshold = 5
                                    print(f"  Difference: {diff_percent:.1f}%")
                                    if diff_percent > threshold:
                                        print(f"  WARNING: Exceeds {threshold}% threshold - will return empty list")
                        
                        # Check for character count mismatch (always check, not just in debug) - AFTER printing debug info
                        if total_comment_chars > 0:
                            diff_percent = abs(total_masked_chars - total_comment_chars) / total_comment_chars * 100
                            threshold = 5
                            if diff_percent > threshold:
                                print(f"WARNING: Character count mismatch in create command! Masked tokens differ from identified comments by {diff_percent:.1f}%")
                                print(f"  Comment characters: {total_comment_chars}, Masked characters: {total_masked_chars}")
                                print(f"  Returning empty list due to excessive mismatch (>{threshold}%)")
                                return []
                        
                        return masked_indices
                    else:
                        # No tokens were masked but we had comments - this is a problem
                        if comment_lines and total_comment_chars > 0:
                            print(f"WARNING: Found {len(comment_lines)} comment lines ({total_comment_chars} chars) in create command but could not mask any tokens!")
                            if DEBUG_MODE:
                                print(f"Comment lines that were not masked:")
                                for line_num in sorted(comment_lines):
                                    if 1 <= line_num <= len(file_lines):
                                        print(f"  Line {line_num}: {repr(file_lines[line_num - 1])}")
                            return []
                        elif DEBUG_MODE:
                            print(f"Could not map comment lines to token indices")
                else:
                    if DEBUG_MODE:
                        print(f"No comment lines found in file")
                
                # Return empty list if no comments found or couldn't map them
                return []
            else:
                if DEBUG_MODE:
                    print(f"StrReplaceEditorProcessor: Create command found but no file_text parameter (assistant turn {assistant_turn})")
                return []
        
        # Handle str_replace command
        elif is_str_replace:
            old_str = self._extract_parameter(content, 'old_str')
            new_str = self._extract_parameter(content, 'new_str')
            
            if old_str is not None or new_str is not None:
                if DEBUG_MODE:
                    print(f"StrReplaceEditorProcessor: Found str_replace command (assistant turn {assistant_turn})")
                    if old_str is not None:
                        old_str_stripped = self._strip_newlines(old_str)
                        print("Full old_str content:")
                        print(old_str_stripped)
                        print("=" * 80)
                    if new_str is not None:
                        new_str_stripped = self._strip_newlines(new_str)
                        print("Full new_str content:")
                        print(new_str_stripped)
                        print("=" * 80)
                
                all_masked_indices = []
                
                # Process old_str if present
                if old_str is not None:
                    old_str_stripped = self._strip_newlines(old_str)
                    comment_lines = self._get_all_comment_lines(old_str_stripped)
                    
                    if comment_lines:
                        if DEBUG_MODE:
                            print(f"Found {len(comment_lines)} comment lines in old_str")
                            # Print the actual comment lines
                            file_lines = old_str_stripped.split('\n')
                            print(f"Comment lines being masked in old_str:")
                            for line_num in sorted(comment_lines):
                                if 1 <= line_num <= len(file_lines):
                                    comment_content = file_lines[line_num - 1]
                                    if comment_content or comment_content.strip():
                                        print(f"  Line {line_num}: {repr(comment_content)}")
                        
                        # Find old_str boundaries using the same method as file_text
                        boundaries = self._find_parameter_boundaries(tokenizer, token_ids, 'old_str')
                        
                        if boundaries is None:
                            print(f"WARNING: Could not find str_replace old_str content in token sequence")
                            return []
                        else:
                            old_str_start_idx, old_str_end_idx = boundaries
                            
                            if DEBUG_MODE:
                                print(f"Found old_str boundaries: start={old_str_start_idx}, end={old_str_end_idx}")
                            
                            # Extract only the tokens that are part of old_str
                            old_str_token_ids = token_ids[old_str_start_idx:old_str_end_idx]
                            
                            # Find token indices corresponding to comment lines within old_str
                            masked_indices = self._find_token_indices_for_lines(
                                tokenizer, old_str_token_ids, content, comment_lines, old_str_stripped, 
                                offset=old_str_start_idx
                            )
                            
                            # Calculate character counts for validation
                            total_comment_chars = 0
                            file_lines = old_str_stripped.split('\n')
                            for line_num in sorted(comment_lines):
                                if 1 <= line_num <= len(file_lines):
                                    total_comment_chars += len(file_lines[line_num - 1]) + 1
                            
                            if masked_indices:
                                # Calculate total characters in masked tokens
                                tokens = tokenizer.convert_ids_to_tokens(token_ids)
                                total_masked_chars = 0
                                for idx in masked_indices:
                                    if 0 <= idx < len(tokens):
                                        token_text = tokens[idx]
                                        clean = token_text.replace('Ġ', ' ').replace('Ċ', '\n').replace('▁', ' ')
                                        total_masked_chars += len(clean)
                                
                                if DEBUG_MODE:
                                    print(f"Masking {len(masked_indices)} tokens for comment lines in old_str")
                                    # Print the actual tokens being masked
                                    print(f"Tokens being masked in old_str:")
                                    
                                    # Group consecutive tokens
                                    if masked_indices:
                                        consecutive_groups = []
                                        current_group = [masked_indices[0]]
                                        
                                        for i in range(1, len(masked_indices)):
                                            if masked_indices[i] == masked_indices[i-1] + 1:
                                                # Consecutive token
                                                current_group.append(masked_indices[i])
                                            else:
                                                # New group
                                                consecutive_groups.append(current_group)
                                                current_group = [masked_indices[i]]
                                        
                                        # Add the last group
                                        consecutive_groups.append(current_group)
                                        
                                        # Print each group
                                        for group in consecutive_groups:
                                            # Build the text for this group
                                            group_text = ""
                                            for idx in group:
                                                if 0 <= idx < len(tokens):
                                                    token_text = tokens[idx]
                                                    # Convert to readable format
                                                    clean = token_text.replace('Ġ', ' ').replace('Ċ', '\n').replace('▁', ' ')
                                                    group_text += clean
                                            
                                            # Print with token range
                                            if len(group) == 1:
                                                print(f"  Token {group[0]}: {repr(group_text)}")
                                            else:
                                                print(f"  Tokens {group[0]}-{group[-1]}: {repr(group_text)}")
                                        
                                        # Show character count comparison in debug mode
                                        print(f"\nCharacter count comparison for old_str:")
                                        print(f"  Total characters in identified comments: {total_comment_chars}")
                                        print(f"  Total characters in masked tokens: {total_masked_chars}")
                                        
                                        if total_comment_chars > 0:
                                            diff_percent = abs(total_masked_chars - total_comment_chars) / total_comment_chars * 100
                                            threshold = 5
                                            print(f"  Difference: {diff_percent:.1f}%")
                                            if diff_percent > threshold:
                                                print(f"  WARNING: Exceeds {threshold}% threshold - will return empty list")
                                
                                # Check for character count mismatch (always check, not just in debug) - AFTER printing debug info
                                if total_comment_chars > 0:
                                    diff_percent = abs(total_masked_chars - total_comment_chars) / total_comment_chars * 100
                                    threshold = 5
                                    if diff_percent > threshold:
                                        print(f"WARNING: Character count mismatch in str_replace old_str! Masked tokens differ from identified comments by {diff_percent:.1f}%")
                                        print(f"  Comment characters: {total_comment_chars}, Masked characters: {total_masked_chars}")
                                        print(f"  Returning empty list due to excessive mismatch (>{threshold}%)")
                                        return []
                                
                                all_masked_indices.extend(masked_indices)
                            else:
                                # No tokens were masked but we had comments - this is a problem
                                if comment_lines and total_comment_chars > 0:
                                    print(f"WARNING: Found {len(comment_lines)} comment lines ({total_comment_chars} chars) in str_replace old_str but could not mask any tokens!")
                                    if DEBUG_MODE:
                                        print(f"Comment lines in old_str that were not masked:")
                                        for line_num in sorted(comment_lines):
                                            if 1 <= line_num <= len(file_lines):
                                                print(f"  Line {line_num}: {repr(file_lines[line_num - 1])}")
                                    return []
                
                # Process new_str if present
                if new_str is not None:
                    new_str_stripped = self._strip_newlines(new_str)
                    comment_lines = self._get_all_comment_lines(new_str_stripped)
                    
                    if comment_lines:
                        if DEBUG_MODE:
                            print(f"Found {len(comment_lines)} comment lines in new_str")
                            # Print the actual comment lines
                            file_lines = new_str_stripped.split('\n')
                            print(f"Comment lines being masked in new_str:")
                            for line_num in sorted(comment_lines):
                                if 1 <= line_num <= len(file_lines):
                                    comment_content = file_lines[line_num - 1]
                                    if comment_content or comment_content.strip():
                                        print(f"  Line {line_num}: {repr(comment_content)}")
                        
                        # Find new_str boundaries using the same method as file_text
                        boundaries = self._find_parameter_boundaries(tokenizer, token_ids, 'new_str')
                        
                        if boundaries is None:
                            print(f"WARNING: Could not find str_replace new_str content in token sequence")
                            return []
                        else:
                            new_str_start_idx, new_str_end_idx = boundaries
                            
                            if DEBUG_MODE:
                                print(f"Found new_str boundaries: start={new_str_start_idx}, end={new_str_end_idx}")
                            
                            # Extract only the tokens that are part of new_str
                            new_str_token_ids = token_ids[new_str_start_idx:new_str_end_idx]
                            
                            # Find token indices corresponding to comment lines within new_str
                            masked_indices = self._find_token_indices_for_lines(
                                tokenizer, new_str_token_ids, content, comment_lines, new_str_stripped, 
                                offset=new_str_start_idx
                            )
                            
                            # Calculate character counts for validation
                            total_comment_chars = 0
                            file_lines = new_str_stripped.split('\n')
                            for line_num in sorted(comment_lines):
                                if 1 <= line_num <= len(file_lines):
                                    total_comment_chars += len(file_lines[line_num - 1]) + 1
                            
                            if masked_indices:
                                # Calculate total characters in masked tokens
                                tokens = tokenizer.convert_ids_to_tokens(token_ids)
                                total_masked_chars = 0
                                for idx in masked_indices:
                                    if 0 <= idx < len(tokens):
                                        token_text = tokens[idx]
                                        clean = token_text.replace('Ġ', ' ').replace('Ċ', '\n').replace('▁', ' ')
                                        total_masked_chars += len(clean)
                                
                                if DEBUG_MODE:
                                    print(f"Masking {len(masked_indices)} tokens for comment lines in new_str")
                                    # Print the actual tokens being masked
                                    print(f"Tokens being masked in new_str:")
                                    
                                    # Group consecutive tokens
                                    if masked_indices:
                                        consecutive_groups = []
                                        current_group = [masked_indices[0]]
                                        
                                        for i in range(1, len(masked_indices)):
                                            if masked_indices[i] == masked_indices[i-1] + 1:
                                                # Consecutive token
                                                current_group.append(masked_indices[i])
                                            else:
                                                # New group
                                                consecutive_groups.append(current_group)
                                                current_group = [masked_indices[i]]
                                        
                                        # Add the last group
                                        consecutive_groups.append(current_group)
                                        
                                        # Print each group
                                        for group in consecutive_groups:
                                            # Build the text for this group
                                            group_text = ""
                                            for idx in group:
                                                if 0 <= idx < len(tokens):
                                                    token_text = tokens[idx]
                                                    # Convert to readable format
                                                    clean = token_text.replace('Ġ', ' ').replace('Ċ', '\n').replace('▁', ' ')
                                                    group_text += clean
                                            
                                            # Print with token range
                                            if len(group) == 1:
                                                print(f"  Token {group[0]}: {repr(group_text)}")
                                            else:
                                                print(f"  Tokens {group[0]}-{group[-1]}: {repr(group_text)}")
                                        
                                        # Show character count comparison in debug mode
                                        print(f"\nCharacter count comparison for new_str:")
                                        print(f"  Total characters in identified comments: {total_comment_chars}")
                                        print(f"  Total characters in masked tokens: {total_masked_chars}")
                                        
                                        if total_comment_chars > 0:
                                            diff_percent = abs(total_masked_chars - total_comment_chars) / total_comment_chars * 100
                                            threshold = 5
                                            print(f"  Difference: {diff_percent:.1f}%")
                                            if diff_percent > threshold:
                                                print(f"  WARNING: Exceeds {threshold}% threshold - will return empty list")
                                
                                # Check for character count mismatch (always check, not just in debug) - AFTER printing debug info
                                if total_comment_chars > 0:
                                    diff_percent = abs(total_masked_chars - total_comment_chars) / total_comment_chars * 100
                                    threshold = 5
                                    if diff_percent > threshold:
                                        print(f"WARNING: Character count mismatch in str_replace new_str! Masked tokens differ from identified comments by {diff_percent:.1f}%")
                                        print(f"  Comment characters: {total_comment_chars}, Masked characters: {total_masked_chars}")
                                        print(f"  Returning empty list due to excessive mismatch (>{threshold}%)")
                                        return []
                                
                                all_masked_indices.extend(masked_indices)
                            else:
                                # No tokens were masked but we had comments - this is a problem
                                if comment_lines and total_comment_chars > 0:
                                    print(f"WARNING: Found {len(comment_lines)} comment lines ({total_comment_chars} chars) in str_replace new_str but could not mask any tokens!")
                                    if DEBUG_MODE:
                                        print(f"Comment lines in new_str that were not masked:")
                                        for line_num in sorted(comment_lines):
                                            if 1 <= line_num <= len(file_lines):
                                                print(f"  Line {line_num}: {repr(file_lines[line_num - 1])}")
                                    return []
                
                # Remove duplicates and sort
                all_masked_indices = sorted(set(all_masked_indices))
                
                if DEBUG_MODE and all_masked_indices:
                    print(f"Total masked indices for str_replace: {len(all_masked_indices)}")
                
                return all_masked_indices
            else:
                if DEBUG_MODE:
                    print(f"StrReplaceEditorProcessor: str_replace command found but no old_str or new_str parameters (assistant turn {assistant_turn})")
                return []
        
        # Return None to continue processing
        return None


class DiffProcessor(LossMaskProcessor):
    """
    Processor that masks diff tokens in str_replace commands.
    
    This processor identifies removed lines from old_str and added lines from new_str
    in str_replace commands and masks their corresponding tokens.
    """
    
    def __init__(self):
        super().__init__()
        self.mask_value = 3  # Set mask value to 2 for this processor
    
    def _extract_parameter(self, content: str, param_name: str) -> Optional[str]:
        """Extract parameter value from XML content."""
        param_tag = f'<parameter={param_name}>'
        if param_tag not in content:
            return None
            
        param_start = content.find(param_tag) + len(param_tag)
        param_end = content.find('</parameter>', param_start)
        
        if param_end == -1:
            return None
            
        return content[param_start:param_end]
    
    def _strip_newlines(self, text: str) -> str:
        """Strip leading and trailing newlines from text."""
        if text.startswith('\n'):
            text = text[1:]
        if text.endswith('\n'):
            text = text[:-1]
        return text
    
    def _check_command_type(self, content: str, command: str) -> bool:
        """Check if content contains a command parameter with optional newlines."""
        # Check all possible variations with newlines
        variations = [
            f'<parameter=command>{command}</parameter>',
            f'<parameter=command>\n{command}</parameter>',
            f'<parameter=command>{command}\n</parameter>',
            f'<parameter=command>\n{command}\n</parameter>'
        ]
        return any(var in content for var in variations)
    
    def _get_diff_lines(self, old_str: str, new_str: str) -> tuple[Set[int], Set[int]]:
        """
        Find the diff between old_str and new_str.
        
        Returns:
            A tuple of (removed_line_nums, added_line_nums) where:
            - removed_line_nums: Line numbers in old_str that were removed
            - added_line_nums: Line numbers in new_str that were added
        """
        import difflib
        
        old_lines = old_str.split('\n')
        new_lines = new_str.split('\n')
        
        # Get the sequence matcher
        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
        
        removed_lines = set()
        added_lines = set()
        
        # Get the opcodes
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'delete':
                # Lines removed from old_str
                for i in range(i1, i2):
                    removed_lines.add(i + 1)  # 1-based line numbers
            elif tag == 'insert':
                # Lines added to new_str
                for j in range(j1, j2):
                    added_lines.add(j + 1)  # 1-based line numbers
            elif tag == 'replace':
                # Lines changed - mark old lines as removed and new lines as added
                for i in range(i1, i2):
                    removed_lines.add(i + 1)
                for j in range(j1, j2):
                    added_lines.add(j + 1)
        
        return removed_lines, added_lines
    
    def _group_consecutive_lines(self, line_nums: Set[int]) -> List[List[int]]:
        """Group consecutive line numbers into blocks."""
        if not line_nums:
            return []
        
        sorted_lines = sorted(line_nums)
        groups = []
        current_group = [sorted_lines[0]]
        
        for i in range(1, len(sorted_lines)):
            if sorted_lines[i] == current_group[-1] + 1:
                current_group.append(sorted_lines[i])
            else:
                groups.append(current_group)
                current_group = [sorted_lines[i]]
        
        groups.append(current_group)
        return groups
    
    def _find_block_in_content(self, content: str, block_text: str, param_name: str, 
                               param_content: str, line_numbers: List[int]) -> Optional[int]:
        """
        Find the character index of a block of text within the full content.
        
        Args:
            content: The full message content
            block_text: The text block to find
            param_name: Parameter name ('old_str' or 'new_str')
            param_content: The full parameter content (old_str or new_str)
            line_numbers: Line numbers of this block in the parameter
            
        Returns:
            Character index where the block starts in content, or None if not found
        """
        # Find the parameter boundaries
        param_tag_start = f'<parameter={param_name}>'
        param_tag_end = '</parameter>'
        
        param_start_pos = content.find(param_tag_start)
        if param_start_pos == -1:
            return None
            
        # Find the end of this parameter
        search_from = param_start_pos + len(param_tag_start)
        param_end_pos = content.find(param_tag_end, search_from)
        if param_end_pos == -1:
            return None
        
        # Extract the parameter content from the original content
        param_in_content = content[search_from:param_end_pos]
        
        # Find the block text within this parameter content
        block_pos = param_in_content.find(block_text)
        if block_pos == -1:
            # Try to find just the first line if full block fails
            first_line = block_text.split('\n')[0]
            block_pos = param_in_content.find(first_line)
            if block_pos == -1:
                return None
            
            if DEBUG_MODE:
                print(f"    Note: Full block not found, using first line position")
        
        # Return the absolute position in content
        return search_from + block_pos
    
    def _is_source_file(self, content: str) -> bool:
        """Check if the path parameter refers to a source file.

        Extracts the path parameter from the XML content, splits it on '/'
        and checks whether any component is an exact match for a keyword in
        SOURCE_FILE_KEYWORDS.

        Returns:
            True if any path component exactly matches a source file keyword,
            False otherwise.
        """
        path = self._extract_parameter(content, 'path')
        if path is None:
            return False
        path_parts = set(path.strip().split('/'))
        return any(keyword in path_parts for keyword in SOURCE_FILE_KEYWORDS)

    def process(self, tokenizer, token_ids: List[int], msg_dict: Dict[str, Any],
                assistant_turn: int) -> List[int]:
        """
        Process str_replace commands and return indices of diff tokens to mask.

        Args:
            tokenizer: The tokenizer instance
            token_ids: List of token IDs for the message
            msg_dict: Dictionary containing 'role' and 'content' fields
            assistant_turn: The turn number for assistant messages

        Returns:
            List of token indices that should be masked (diff tokens)
        """
        # Skip non-assistant messages
        if msg_dict.get("role") != "assistant":
            return []

        content = msg_dict.get("content", "")

        # Check if content contains str_replace_editor function call
        if '<function=str_replace_editor>' not in content:
            return []

        # Check if it's a str_replace command
        is_str_replace = False

        if self._check_command_type(content, 'str_replace'):
            is_str_replace = True
        elif '<parameter=old_str>' in content:
            is_str_replace = True

        if not is_str_replace:
            return []

        # Only process str_replace actions on source files
        if not self._is_source_file(content):
            return []
        if DEBUG_MODE:
            source_path = self._extract_parameter(content, 'path')
            print(f"DiffProcessor: processing str_replace on source file: {source_path.strip() if source_path else 'unknown'}")
        
        # Extract old_str and new_str
        old_str = self._extract_parameter(content, 'old_str')
        new_str = self._extract_parameter(content, 'new_str')
        
        if old_str is None or new_str is None:
            print(f"WARNING: DiffProcessor could not extract old_str or new_str from str_replace command (assistant turn {assistant_turn})")
            return []
        
        # Strip newlines
        old_str_stripped = self._strip_newlines(old_str)
        new_str_stripped = self._strip_newlines(new_str)
        
        if DEBUG_MODE:
            print(f"DiffProcessor: Processing str_replace command (assistant turn {assistant_turn})")
        
        # Get diff lines
        removed_lines, added_lines = self._get_diff_lines(old_str_stripped, new_str_stripped)
        
        # Group consecutive lines
        removed_blocks = self._group_consecutive_lines(removed_lines)
        added_blocks = self._group_consecutive_lines(added_lines)
        
        diff_blocks = []
        
        # Process removed blocks from old_str
        old_lines = old_str_stripped.split('\n')
        for block in removed_blocks:
            # Build the block text
            block_lines = []
            for line_num in block:
                if 1 <= line_num <= len(old_lines):
                    block_lines.append(old_lines[line_num - 1])
            
            if block_lines:
                block_text = '\n'.join(block_lines)
                char_idx = self._find_block_in_content(content, block_text, 'old_str', 
                                                       old_str_stripped, block)
                if char_idx is not None:
                    diff_blocks.append({
                        'text': block_text,
                        'char_index': char_idx,
                        'lines': block,
                        'param': 'old_str',
                        'type': 'removed'
                    })
                else:
                    print(f"WARNING: DiffProcessor could not find character index for removed block lines {block} in old_str")
                    if DEBUG_MODE and block_lines:
                        print(f"  Block text: {repr(block_text)}")
        
        # Process added blocks from new_str
        new_lines = new_str_stripped.split('\n')
        for block in added_blocks:
            # Build the block text
            block_lines = []
            for line_num in block:
                if 1 <= line_num <= len(new_lines):
                    block_lines.append(new_lines[line_num - 1])
            
            if block_lines:
                block_text = '\n'.join(block_lines)
                char_idx = self._find_block_in_content(content, block_text, 'new_str', 
                                                       new_str_stripped, block)
                if char_idx is not None:
                    diff_blocks.append({
                        'text': block_text,
                        'char_index': char_idx,
                        'lines': block,
                        'param': 'new_str',
                        'type': 'added'
                    })
                else:
                    print(f"WARNING: DiffProcessor could not find character index for added block lines {block} in new_str")
                    if DEBUG_MODE and block_lines:
                        print(f"  Block text: {repr(block_text)}")
        
        if DEBUG_MODE:
            print(f"DiffProcessor: Found {len(removed_blocks)} removed blocks and {len(added_blocks)} added blocks")
            for block_info in diff_blocks:
                print(f"  {block_info['type']} block in {block_info['param']} at char {block_info['char_index']}: lines {block_info['lines']}")
                print(f"    Text: {repr(block_info['text'])}")
        
        # Map character indices to token indices
        masked_indices = []
        
        if not diff_blocks:
            if DEBUG_MODE:
                print(f"DiffProcessor: No diff blocks found to map to tokens")
            return []
        
        # First, find where the content starts in the token sequence
        # Take first characters of content as a search pattern, but skip leading whitespace
        content_stripped = content.lstrip()
        search_pattern = content_stripped[:50] if len(content_stripped) >= 50 else content_stripped
        content_start_char = -1
        
        # Decode the full token sequence
        try:
            decoded_full = tokenizer.decode(token_ids, skip_special_tokens=False)
            
            # Find where our content starts
            content_start_char = decoded_full.find(search_pattern)
            if content_start_char == -1:
                print(f"WARNING: DiffProcessor could not find content start in token sequence")
                print(f"  Search pattern: {repr(search_pattern[:30])}...")
                return []
            
            # Adjust for any leading whitespace we skipped
            whitespace_skipped = len(content) - len(content_stripped)
            content_start_char = content_start_char - whitespace_skipped
            
            if DEBUG_MODE:
                print(f"DiffProcessor: Found content start at character position {content_start_char} in decoded tokens")
                
        except Exception as e:
            print(f"WARNING: DiffProcessor error finding content start: {str(e)}")
            return []
        
        # Build accurate character position to token mapping using actual decoded positions
        token_char_ranges = []  # List of (start_char, end_char, token_idx)
        
        try:
            for idx in range(len(token_ids)):
                # Get the text up to this token by decoding
                if idx == 0:
                    prefix_text = ""
                else:
                    prefix_text = tokenizer.decode(token_ids[:idx], skip_special_tokens=False)
                
                # Get the text including this token
                current_text = tokenizer.decode(token_ids[:idx+1], skip_special_tokens=False)
                
                # The token spans from end of prefix to end of current
                start_char = len(prefix_text)
                end_char = len(current_text)
                token_char_ranges.append((start_char, end_char, idx))
        except Exception as e:
            print(f"WARNING: DiffProcessor error building token character mapping: {str(e)}")
            return []
        
        if DEBUG_MODE:
            total_chars = len(decoded_full) if token_char_ranges else 0
            print(f"DiffProcessor: Total characters in token sequence: {total_chars}")
            print(f"DiffProcessor: Mapping {len(diff_blocks)} diff blocks to tokens")
        
        # For each diff block, find the tokens it spans
        for block_info in diff_blocks:
            # Adjust block positions by the content start offset
            adjusted_block_start = block_info['char_index'] + content_start_char
            adjusted_block_end = adjusted_block_start + len(block_info['text'])
            
            if DEBUG_MODE:
                print(f"  Block char range in content: {block_info['char_index']}-{block_info['char_index'] + len(block_info['text'])}")
                print(f"  Adjusted char range in tokens: {adjusted_block_start}-{adjusted_block_end}")
                # Show what's actually at that position in the decoded content
                decoded_full = tokenizer.decode(token_ids, skip_special_tokens=False)
                sample_start = max(0, adjusted_block_start - 10)
                sample_end = min(len(decoded_full), adjusted_block_start + 50)
                print(f"  Content at adjusted position: {repr(decoded_full[sample_start:sample_end])}")
            
            block_token_indices = []
            
            # Find tokens that overlap with this block
            for start_char, end_char, token_idx in token_char_ranges:
                # Check if this token overlaps with the block
                if start_char < adjusted_block_end and end_char > adjusted_block_start:
                    block_token_indices.append(token_idx)
            
            if block_token_indices:
                if DEBUG_MODE:
                    print(f"  Block '{block_info['type']}' lines {block_info['lines']} maps to tokens {block_token_indices[0]}-{block_token_indices[-1]}")
                    # Print the actual masked content
                    masked_content = []
                    for token_idx in block_token_indices:
                        if token_idx < len(token_ids):
                            token_str = tokenizer.convert_ids_to_tokens([token_ids[token_idx]])[0]
                            # Normalize for display
                            normalized = token_str.replace('Ġ', ' ').replace('▁', ' ').replace('Ċ', '\n')
                            masked_content.append(normalized)
                    masked_text = ''.join(masked_content)
                    # Apply Unicode correction for display
                    display_text = apply_unicode_corrections(masked_text)
                    print(f"    Masked content: {repr(display_text)}")
                    
                    # Verify the masked content matches the expected block text
                    expected_text = block_info['text']
                    if masked_text != expected_text:
                        # Check various acceptable differences
                        is_acceptable = False
                        
                        # 1. Check if it's just a whitespace difference
                        if masked_text.strip() == expected_text.strip():
                            is_acceptable = True
                        
                        # 2. Check if expected_text is a substring of masked_text with small length difference
                        elif expected_text in masked_text and abs(len(masked_text) - len(expected_text)) <= 10:
                            is_acceptable = True
                        
                        # 3. Check if masked_text is a substring of expected_text with small length difference
                        elif masked_text in expected_text and abs(len(masked_text) - len(expected_text)) <= 10:
                            is_acceptable = True
                        
                        # 4. Check for Unicode corruption issues - compare after removing potential corrupted chars
                        elif abs(len(masked_text) - len(expected_text)) <= 10:
                            # Apply Unicode corruption corrections
                            masked_normalized = apply_unicode_corrections(masked_text)
                            if masked_normalized.strip() == expected_text.strip():
                                is_acceptable = True
                            # Also try with leading/trailing character differences
                            elif len(masked_normalized) > len(expected_text) and expected_text in masked_normalized:
                                is_acceptable = True
                            elif len(expected_text) > len(masked_normalized) and masked_normalized in expected_text:
                                is_acceptable = True
                        
                        if not is_acceptable:
                            # Return empty list when validation fails
                            if DEBUG_MODE:
                                print(f"    DEBUG: Masked content validation failed, returning empty mask")
                                print(f"    Expected: {repr(expected_text)}")
                                print(f"    Actually masked: {repr(apply_unicode_corrections(masked_text))}")
                            return []
                # Also verify content when not in debug mode (but without printing everything)
                if not DEBUG_MODE:
                    # Quick verification
                    masked_content = []
                    for token_idx in block_token_indices:
                        if token_idx < len(token_ids):
                            token_str = tokenizer.convert_ids_to_tokens([token_ids[token_idx]])[0]
                            normalized = token_str.replace('Ġ', ' ').replace('▁', ' ').replace('Ċ', '\n')
                            masked_content.append(normalized)
                    masked_text = ''.join(masked_content)
                    expected_text = block_info['text']
                    
                    # Check if the mismatch is acceptable
                    is_acceptable = (
                        # Whitespace-only difference
                        masked_text.strip() == expected_text.strip() or
                        # Expected is substring of masked with small difference
                        (expected_text in masked_text and abs(len(masked_text) - len(expected_text)) <= 10) or
                        # Masked is substring of expected with small difference  
                        (masked_text in expected_text and abs(len(masked_text) - len(expected_text)) <= 10) or
                        # Handle Unicode corruption
                        (abs(len(masked_text) - len(expected_text)) <= 10 and 
                         (apply_unicode_corrections(masked_text).strip() == expected_text.strip() or
                          expected_text in apply_unicode_corrections(masked_text) or
                          apply_unicode_corrections(masked_text) in expected_text))
                    )
                    
                    if not is_acceptable:
                        # Return empty list when validation fails
                        return []
                
                masked_indices.extend(block_token_indices)
            else:
                # Return empty list if block has no matching tokens
                return []
        
        # Remove duplicates and sort
        masked_indices = sorted(set(masked_indices))
        
        if DEBUG_MODE:
            print(f"DiffProcessor: Total tokens to mask: {len(masked_indices)}")
        
        return masked_indices


class CommandGtProcessor(LossMaskProcessor):
    """
    Original processor that masks '>' tokens when preceded by 'command'.
    
    This is the existing logic from get_zero_loss_mask_indices.
    """
    
    def __init__(self):
        super().__init__()
        self.mask_value = 0  # Set mask value to 0 for this processor
    
    def process(self, tokenizer, token_ids: List[int], msg_dict: Dict[str, Any], 
                assistant_turn: int) -> List[int]:
        """
        Dummy implementation - always returns empty list.
        
        This processor is a placeholder for future implementation
        of masking '>' tokens preceded by 'command'.
        
        Args:
            tokenizer: The tokenizer instance
            token_ids: List of token IDs for the message
            msg_dict: Dictionary containing 'role' and 'content' fields
            assistant_turn: The turn number for assistant messages
            
        Returns:
            Empty list (no masking applied)
        """
        # Always return empty list - dummy implementation for now
        return []


class LossMaskProcessorManager:
    """Manager class that handles multiple loss mask processors."""
    
    def __init__(self):
        """Initialize the processor manager without any processors."""
        self.processors: List[LossMaskProcessor] = []
        # Store all available processors in registered order
        self.available_processors = {
            'StrReplaceEditorProcessor': StrReplaceEditorProcessor,
            'DiffProcessor': DiffProcessor,
            'CommandGtProcessor': CommandGtProcessor
        }
        # Define the order in which processors should be applied
        # Later processors overwrite earlier ones for overlapping indices
        self.processor_order = ['StrReplaceEditorProcessor', 'DiffProcessor', 'CommandGtProcessor']
    
    def configure_processors(self, processor_names: Optional[str]):
        """Configure which processors to use based on command line argument.
        
        Args:
            processor_names: None (no processors), "all" (all processors), 
                           or comma-separated list of processor names
        """
        self.processors.clear()
        
        if processor_names is None:
            # No processors - default behavior
            return
        
        if processor_names.lower() == 'all':
            # Use all processors in registered order
            for proc_name in self.processor_order:
                if proc_name in self.available_processors:
                    self.processors.append(self.available_processors[proc_name]())
        else:
            # Use specific processors in registered order
            requested = [name.strip() for name in processor_names.split(',')]
            # Apply in registered order, not command line order
            for proc_name in self.processor_order:
                if proc_name in requested and proc_name in self.available_processors:
                    self.processors.append(self.available_processors[proc_name]())
        
        if DEBUG_MODE:
            print(f"Configured processors: {[type(p).__name__ for p in self.processors]}")
    
    def add_processor(self, processor: LossMaskProcessor):
        """Add a new processor to the manager."""
        self.processors.append(processor)
    
    def get_zero_loss_mask_indices(self, tokenizer, token_ids: List[int], 
                                   msg_dict: Dict[str, Any], assistant_turn: int) -> List[int]:
        """
        Process the message through all processors to get loss mask array.
        
        This is the main entry point that replaces the original get_zero_loss_mask_indices function.
        
        Args:
            tokenizer: The tokenizer instance
            token_ids: List of token IDs for the message
            msg_dict: Dictionary containing 'role' and 'content' fields
            assistant_turn: The turn number for assistant messages
            
        Returns:
            List of same length as token_ids where:
            - processor.mask_value at indices that should be masked
            - -1 at all other indices
        """
        # Initialize mask array with -1 (no mask)
        mask_array = [-1] * len(token_ids)
        
        # Process through each processor in order
        # Later processors overwrite earlier ones for overlapping indices
        for processor in self.processors:
            result = processor.process(tokenizer, token_ids, msg_dict, assistant_turn)
            
            # If a processor returns indices, update the mask array
            if result is not None and isinstance(result, list) and len(result) > 0:
                for idx in result:
                    if 0 <= idx < len(token_ids):
                        mask_array[idx] = processor.mask_value
        
        return mask_array


# Global instance for easy access
loss_mask_processor_manager = LossMaskProcessorManager()


def configure_loss_mask_processors(processor_names: Optional[str]):
    """Configure which loss mask processors to use.
    
    Args:
        processor_names: None (no processors), "all" (all processors), 
                       or comma-separated list of processor names
    """
    loss_mask_processor_manager.configure_processors(processor_names)


def get_zero_loss_mask_indices(tokenizer, token_ids: List[int], 
                               msg_dict: Dict[str, Any], assistant_turn: int) -> List[int]:
    """
    Main function to get loss mask array.
    
    This function serves as the single entry point from create_weighted_sft_dataset.py
    and internally handles all processors.
    
    Args:
        tokenizer: The tokenizer instance
        token_ids: List of token IDs for the message
        msg_dict: Dictionary containing 'role' and 'content' fields
        assistant_turn: The turn number for assistant messages
        
    Returns:
        List of same length as token_ids where:
        - 0 at indices that should be masked (loss = 0)
        - -1 at all other indices
    """
    return loss_mask_processor_manager.get_zero_loss_mask_indices(
        tokenizer, token_ids, msg_dict, assistant_turn
    )