"""
Enhanced DeepSeek response parsing system.

This module provides robust parsing capabilities for DeepSeek R1 responses,
including reasoning content extraction from <think> tags and comprehensive
JSON parsing with detailed error reporting.
"""

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

__all__ = [
    "ParsedResponse",
    "ReasoningContent", 
    "ParseError",
    "ParseErrorType",
    "DeepSeekResponseParser"
]

logger = logging.getLogger(__name__)


class ParseErrorType(str, Enum):
    """Types of parsing errors that can occur."""
    JSON_SYNTAX_ERROR = "json_syntax_error"
    JSON_STRUCTURE_ERROR = "json_structure_error"
    REASONING_EXTRACTION_ERROR = "reasoning_extraction_error"
    CONTENT_VALIDATION_ERROR = "content_validation_error"
    ENCODING_ERROR = "encoding_error"
    UNEXPECTED_FORMAT = "unexpected_format"


@dataclass
class ParseError:
    """Detailed information about a parsing error."""
    error_type: ParseErrorType
    message: str
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    context: Optional[str] = None
    suggestion: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ReasoningContent:
    """Extracted reasoning content from DeepSeek responses."""
    raw_reasoning: str
    structured_steps: List[str]
    confidence_indicators: List[str]
    decision_points: List[str]
    extracted_at: datetime = None
    
    def __post_init__(self):
        if self.extracted_at is None:
            self.extracted_at = datetime.now()


@dataclass
class ParsedResponse:
    """Comprehensive parsed response from DeepSeek."""
    raw_content: str
    main_content: str
    reasoning: Optional[ReasoningContent] = None
    json_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    parse_errors: List[ParseError] = None
    parsing_time: float = 0.0
    parsed_at: datetime = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.parse_errors is None:
            self.parse_errors = []
        if self.parsed_at is None:
            self.parsed_at = datetime.now()
    
    def has_errors(self) -> bool:
        """Check if parsing encountered any errors."""
        return len(self.parse_errors) > 0
    
    def has_reasoning(self) -> bool:
        """Check if reasoning content was extracted."""
        return self.reasoning is not None
    
    def has_json(self) -> bool:
        """Check if JSON data was successfully parsed."""
        return self.json_data is not None


class DeepSeekResponseParser:
    """
    Enhanced parser for DeepSeek R1 responses with robust error handling.
    
    Supports extraction of reasoning content from <think> tags and other formats,
    comprehensive JSON parsing with detailed error reporting, and flexible
    content validation.
    """
    
    def __init__(self, 
                 strict_json: bool = False,
                 extract_reasoning: bool = True,
                 validate_content: bool = True):
        """
        Initialize the response parser.
        
        Args:
            strict_json: Whether to use strict JSON parsing
            extract_reasoning: Whether to extract reasoning content
            validate_content: Whether to validate parsed content
        """
        self.strict_json = strict_json
        self.extract_reasoning = extract_reasoning
        self.validate_content = validate_content
        
        # Compile regex patterns for efficiency
        self._think_pattern = re.compile(
            r'<think>(.*?)</think>', 
            re.DOTALL | re.IGNORECASE
        )
        self._reasoning_pattern = re.compile(
            r'<reasoning>(.*?)</reasoning>', 
            re.DOTALL | re.IGNORECASE
        )
        self._json_pattern = re.compile(
            r'```json\s*(.*?)\s*```', 
            re.DOTALL | re.IGNORECASE
        )
        self._code_block_pattern = re.compile(
            r'```(?:json)?\s*(.*?)\s*```', 
            re.DOTALL | re.IGNORECASE
        )
        
        # Patterns for reasoning analysis
        self._step_pattern = re.compile(
            r'(?:step\s*\d+|first|second|third|next|then|finally)[:\-\s]+(.*?)(?=\n|$)', 
            re.IGNORECASE
        )
        self._confidence_pattern = re.compile(
            r'(?:confident|certain|sure|likely|probably|maybe|uncertain)[^.]*\.', 
            re.IGNORECASE
        )
        self._decision_pattern = re.compile(
            r'(?:decide|choose|select|determine|conclude)[^.]*\.', 
            re.IGNORECASE
        )
    
    def parse_response(self, response_content: str) -> ParsedResponse:
        """
        Parse a DeepSeek response with comprehensive error handling.
        
        Args:
            response_content: Raw response content from DeepSeek
            
        Returns:
            ParsedResponse with extracted content and metadata
        """
        start_time = datetime.now()
        parse_errors = []
        
        try:
            # Clean and normalize input
            cleaned_content = self._clean_response_content(response_content)
            
            # Extract reasoning content if enabled
            reasoning = None
            if self.extract_reasoning:
                reasoning, reasoning_errors = self._extract_reasoning_content(cleaned_content)
                parse_errors.extend(reasoning_errors)
            
            # Extract main content (non-reasoning)
            main_content = self._extract_main_content(cleaned_content)
            
            # Attempt JSON parsing
            json_data, json_errors = self._parse_json_content(cleaned_content)
            parse_errors.extend(json_errors)
            
            # Validate content if enabled
            if self.validate_content:
                validation_errors = self._validate_parsed_content(main_content, json_data, reasoning)
                parse_errors.extend(validation_errors)
            
            # Calculate parsing time
            parsing_time = (datetime.now() - start_time).total_seconds()
            
            # Build metadata
            metadata = {
                "original_length": len(response_content),
                "cleaned_length": len(cleaned_content),
                "has_think_tags": "<think>" in response_content.lower(),
                "has_reasoning_tags": "<reasoning>" in response_content.lower(),
                "has_json_blocks": "```json" in response_content.lower(),
                "parsing_duration_ms": round(parsing_time * 1000, 2)
            }
            
            return ParsedResponse(
                raw_content=response_content,
                main_content=main_content,
                reasoning=reasoning,
                json_data=json_data,
                metadata=metadata,
                parse_errors=parse_errors,
                parsing_time=parsing_time
            )
            
        except Exception as e:
            # Handle unexpected parsing errors
            error = ParseError(
                error_type=ParseErrorType.UNEXPECTED_FORMAT,
                message=f"Unexpected error during parsing: {str(e)}",
                context=response_content[:200] + "..." if len(response_content) > 200 else response_content,
                suggestion="Check response format and try again"
            )
            
            parsing_time = (datetime.now() - start_time).total_seconds()
            
            return ParsedResponse(
                raw_content=response_content,
                main_content=response_content,  # Fallback to raw content
                parse_errors=[error],
                parsing_time=parsing_time,
                metadata={"parsing_failed": True, "error_type": str(type(e).__name__)}
            )    
    def _clean_response_content(self, content: str) -> str:
        """
        Clean and normalize response content.
        
        Args:
            content: Raw response content
            
        Returns:
            Cleaned content string
        """
        if not content:
            return ""
        
        try:
            # Handle encoding issues
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='replace')
            
            # Normalize line endings
            content = content.replace('\r\n', '\n').replace('\r', '\n')
            
            # Remove excessive whitespace while preserving structure
            lines = content.split('\n')
            cleaned_lines = []
            
            for line in lines:
                # Preserve indentation but clean trailing whitespace
                cleaned_line = line.rstrip()
                cleaned_lines.append(cleaned_line)
            
            # Remove excessive empty lines (more than 2 consecutive)
            result_lines = []
            empty_count = 0
            
            for line in cleaned_lines:
                if line.strip() == "":
                    empty_count += 1
                    if empty_count <= 2:  # Allow up to 2 consecutive empty lines
                        result_lines.append(line)
                else:
                    empty_count = 0
                    result_lines.append(line)
            
            return '\n'.join(result_lines)
            
        except Exception as e:
            logger.warning(f"Error cleaning response content: {e}")
            return content  # Return original if cleaning fails
    
    def _extract_reasoning_content(self, content: str) -> Tuple[Optional[ReasoningContent], List[ParseError]]:
        """
        Extract reasoning content from various tag formats.
        
        Args:
            content: Cleaned response content
            
        Returns:
            Tuple of (ReasoningContent or None, list of parse errors)
        """
        errors = []
        
        try:
            # Try to extract from <think> tags first
            think_matches = self._think_pattern.findall(content)
            reasoning_matches = self._reasoning_pattern.findall(content)
            
            raw_reasoning = ""
            
            # Combine all reasoning content
            if think_matches:
                raw_reasoning += "\n".join(think_matches)
            
            if reasoning_matches:
                if raw_reasoning:
                    raw_reasoning += "\n\n"
                raw_reasoning += "\n".join(reasoning_matches)
            
            if not raw_reasoning.strip():
                return None, errors
            
            # Analyze reasoning structure
            structured_steps = self._extract_reasoning_steps(raw_reasoning)
            confidence_indicators = self._extract_confidence_indicators(raw_reasoning)
            decision_points = self._extract_decision_points(raw_reasoning)
            
            reasoning_content = ReasoningContent(
                raw_reasoning=raw_reasoning.strip(),
                structured_steps=structured_steps,
                confidence_indicators=confidence_indicators,
                decision_points=decision_points
            )
            
            return reasoning_content, errors
            
        except Exception as e:
            error = ParseError(
                error_type=ParseErrorType.REASONING_EXTRACTION_ERROR,
                message=f"Failed to extract reasoning content: {str(e)}",
                context=content[:100] + "..." if len(content) > 100 else content,
                suggestion="Check for malformed reasoning tags"
            )
            errors.append(error)
            return None, errors
    
    def _extract_reasoning_steps(self, reasoning_text: str) -> List[str]:
        """Extract structured reasoning steps from text."""
        steps = []
        
        # Find explicit step markers
        step_matches = self._step_pattern.findall(reasoning_text)
        steps.extend([step.strip() for step in step_matches if step.strip()])
        
        # If no explicit steps found, try to identify logical progression
        if not steps:
            sentences = re.split(r'[.!?]+', reasoning_text)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10:  # Filter out very short fragments
                    # Look for reasoning indicators
                    if any(indicator in sentence.lower() for indicator in 
                          ['because', 'since', 'therefore', 'thus', 'so', 'hence']):
                        steps.append(sentence)
        
        return steps[:10]  # Limit to 10 steps for practicality
    
    def _extract_confidence_indicators(self, reasoning_text: str) -> List[str]:
        """Extract confidence indicators from reasoning text."""
        confidence_matches = self._confidence_pattern.findall(reasoning_text)
        return [match.strip() for match in confidence_matches][:5]  # Limit to 5
    
    def _extract_decision_points(self, reasoning_text: str) -> List[str]:
        """Extract decision points from reasoning text."""
        decision_matches = self._decision_pattern.findall(reasoning_text)
        return [match.strip() for match in decision_matches][:5]  # Limit to 5
    
    def _extract_main_content(self, content: str) -> str:
        """
        Extract main content by removing reasoning tags.
        
        Args:
            content: Cleaned response content
            
        Returns:
            Main content without reasoning tags
        """
        # Remove <think> tags and their content
        content_without_think = self._think_pattern.sub('', content)
        
        # Remove <reasoning> tags and their content
        content_without_reasoning = self._reasoning_pattern.sub('', content_without_think)
        
        # Clean up extra whitespace
        lines = content_without_reasoning.split('\n')
        cleaned_lines = [line.rstrip() for line in lines]
        
        # Remove excessive empty lines
        result_lines = []
        empty_count = 0
        
        for line in cleaned_lines:
            if line.strip() == "":
                empty_count += 1
                if empty_count <= 1:  # Allow only 1 consecutive empty line in main content
                    result_lines.append(line)
            else:
                empty_count = 0
                result_lines.append(line)
        
        return '\n'.join(result_lines).strip()    
    def _parse_json_content(self, content: str) -> Tuple[Optional[Dict[str, Any]], List[ParseError]]:
        """
        Parse JSON content with comprehensive error reporting.
        
        Args:
            content: Content to parse for JSON
            
        Returns:
            Tuple of (parsed JSON dict or None, list of parse errors)
        """
        errors = []
        
        # Try different JSON extraction strategies
        json_candidates = []
        
        # Strategy 1: Look for JSON code blocks
        json_block_matches = self._json_pattern.findall(content)
        json_candidates.extend(json_block_matches)
        
        # Strategy 2: Look for any code blocks that might contain JSON
        code_block_matches = self._code_block_pattern.findall(content)
        for match in code_block_matches:
            if match.strip().startswith(('{', '[')):
                json_candidates.append(match)
        
        # Strategy 3: Look for JSON-like structures in the content
        json_like_pattern = re.compile(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', re.DOTALL)
        json_like_matches = json_like_pattern.findall(content)
        json_candidates.extend(json_like_matches)
        
        # Try to parse each candidate
        for i, candidate in enumerate(json_candidates):
            try:
                candidate = candidate.strip()
                if not candidate:
                    continue
                
                # Attempt to parse JSON
                if self.strict_json:
                    parsed_json = json.loads(candidate)
                else:
                    # Try to fix common JSON issues
                    fixed_candidate = self._fix_common_json_issues(candidate)
                    parsed_json = json.loads(fixed_candidate)
                
                # Validate that it's a dict or list
                if isinstance(parsed_json, (dict, list)):
                    return parsed_json, errors
                
            except json.JSONDecodeError as e:
                error = ParseError(
                    error_type=ParseErrorType.JSON_SYNTAX_ERROR,
                    message=f"JSON syntax error in candidate {i+1}: {str(e)}",
                    line_number=getattr(e, 'lineno', None),
                    column_number=getattr(e, 'colno', None),
                    context=candidate[:100] + "..." if len(candidate) > 100 else candidate,
                    suggestion=self._get_json_error_suggestion(e)
                )
                errors.append(error)
                
            except Exception as e:
                error = ParseError(
                    error_type=ParseErrorType.JSON_STRUCTURE_ERROR,
                    message=f"JSON structure error in candidate {i+1}: {str(e)}",
                    context=candidate[:100] + "..." if len(candidate) > 100 else candidate,
                    suggestion="Check JSON structure and data types"
                )
                errors.append(error)
        
        return None, errors
    
    def _fix_common_json_issues(self, json_str: str) -> str:
        """
        Attempt to fix common JSON formatting issues.
        
        Args:
            json_str: JSON string with potential issues
            
        Returns:
            Fixed JSON string
        """
        # Remove trailing commas
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Fix single quotes to double quotes (but preserve strings)
        json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)  # Keys
        json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)  # String values
        
        # Fix unquoted keys
        json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
        
        return json_str
    
    def _get_json_error_suggestion(self, error: json.JSONDecodeError) -> str:
        """
        Get helpful suggestion based on JSON decode error.
        
        Args:
            error: JSONDecodeError instance
            
        Returns:
            Helpful suggestion string
        """
        error_msg = str(error).lower()
        
        if "trailing comma" in error_msg:
            return "Remove trailing commas before closing brackets/braces"
        elif "expecting property name" in error_msg:
            return "Ensure all object keys are properly quoted"
        elif "expecting value" in error_msg:
            return "Check for missing values or extra commas"
        elif "unterminated string" in error_msg:
            return "Check for unescaped quotes in string values"
        elif "expecting ',' delimiter" in error_msg:
            return "Add missing commas between array/object elements"
        else:
            return "Check JSON syntax and structure"
    
    def _validate_parsed_content(self, 
                                main_content: str, 
                                json_data: Optional[Dict[str, Any]], 
                                reasoning: Optional[ReasoningContent]) -> List[ParseError]:
        """
        Validate parsed content for consistency and completeness.
        
        Args:
            main_content: Extracted main content
            json_data: Parsed JSON data
            reasoning: Extracted reasoning content
            
        Returns:
            List of validation errors
        """
        errors = []
        
        try:
            # Check if main content is suspiciously short
            if len(main_content.strip()) < 10:
                error = ParseError(
                    error_type=ParseErrorType.CONTENT_VALIDATION_ERROR,
                    message="Main content appears to be too short",
                    context=main_content,
                    suggestion="Check if reasoning extraction removed too much content"
                )
                errors.append(error)
            
            # Check if JSON data has expected structure (if present)
            if json_data is not None:
                if isinstance(json_data, dict) and not json_data:
                    error = ParseError(
                        error_type=ParseErrorType.CONTENT_VALIDATION_ERROR,
                        message="JSON data is empty dictionary",
                        suggestion="Verify JSON content extraction"
                    )
                    errors.append(error)
                elif isinstance(json_data, list) and not json_data:
                    error = ParseError(
                        error_type=ParseErrorType.CONTENT_VALIDATION_ERROR,
                        message="JSON data is empty list",
                        suggestion="Verify JSON content extraction"
                    )
                    errors.append(error)
            
            # Check reasoning content consistency
            if reasoning is not None:
                if not reasoning.raw_reasoning.strip():
                    error = ParseError(
                        error_type=ParseErrorType.CONTENT_VALIDATION_ERROR,
                        message="Reasoning content is empty",
                        suggestion="Check reasoning extraction logic"
                    )
                    errors.append(error)
                
                # Check if reasoning steps were extracted
                if not reasoning.structured_steps:
                    # This is a warning, not an error
                    logger.debug("No structured reasoning steps found")
            
        except Exception as e:
            error = ParseError(
                error_type=ParseErrorType.CONTENT_VALIDATION_ERROR,
                message=f"Validation error: {str(e)}",
                suggestion="Review content validation logic"
            )
            errors.append(error)
        
        return errors
    
    def get_parser_stats(self) -> Dict[str, Any]:
        """
        Get statistics about parser configuration and capabilities.
        
        Returns:
            Dictionary with parser statistics
        """
        return {
            "strict_json": self.strict_json,
            "extract_reasoning": self.extract_reasoning,
            "validate_content": self.validate_content,
            "supported_reasoning_tags": ["<think>", "<reasoning>"],
            "supported_json_formats": ["```json", "code blocks", "inline JSON"],
            "max_reasoning_steps": 10,
            "max_confidence_indicators": 5,
            "max_decision_points": 5
        }