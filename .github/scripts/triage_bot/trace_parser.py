import csv
import os
import re
import json
from typing import List, Dict, Optional

class TraceParser:
    def __init__(self, traceback_text, work_dir=os.getcwd()):
        self.trace_dict = {}
        self.trace_text = traceback_text
        self.word_dir = work_dir

    def parse_log_traceback(self) -> Dict:
        """
        Parse the self.trace_text into a structured format.
        
        Returns:
            Dictionary with parsed traceback information:
            {
                'exception_type': str,
                'exception_message': str,
                'stack': List[Dict],  # Stack frames
                'full_traceback': str  # Original text
            }
        """
        traceback_text = self.trace_text
        lines = traceback_text.split('\n')
        if not lines:
            return {}
        
        # Extract exception type and message (last line)
        exe_line = ""
        reversed_lines = lines[::-1]
        for l in reversed_lines:
            if ": " in l:
                exc_line = l
                break

        exc_type, exc_message = self.parse_exception_line(exc_line)
        
        # Parse stack frames
        stack = []
        current_frame = None
        
        for line in lines[:-1]:
            line = line.strip()
            
            # Check for new frame (File "...")
            frame_match = re.match(
                r'File "(?P<file>.+)", line (?P<line>\d+), in (?P<function>.+)', 
                line
            )
            
            if frame_match:
                # Save previous frame if exists
                if current_frame:
                    stack.append(current_frame)
                
                # Start new frame
                current_frame = {
                    'file': frame_match.group('file'),
                    'line': int(frame_match.group('line')),
                    'function': frame_match.group('function'),
                    'code': None
                }
            elif current_frame and line and not line.startswith('File "'):
                # This is likely the code line for the current frame
                current_frame['code'] = line
        
        # Add the last frame if exists
        if current_frame:
            stack.append(current_frame)
        
        self.trace_dict = {
            'exception_type': exc_type,
            'exception_message': exc_message,
            'stack': stack,
            'full_traceback': traceback_text
        }

        return self.trace_dict

    def parse_exception_line(self, line: str) -> (str, str):
        """
        Parse the exception type and message from traceback last line.
        
        Handles cases like:
        "ValueError: invalid literal"
        "TypeError: something went wrong (error code: 123)"
        """
        if ': ' in line:
            exc_type, exc_message = line.split(': ', 1)
            return exc_type.strip(), exc_message.strip()
        return line.strip(), ''


fail_list_path = "ut_failure_list.txt"

with open(fail_list_path, 'r') as file:
    for line in file:
        try:
            data = json.loads(line)
            traceback = data['Traceback']
            trace_parser = TraceParser(traceback)
            trace_parser.parse_log_traceback()
            print(trace_parser.trace_dict)
        except:
            print("json load failed!\n")


