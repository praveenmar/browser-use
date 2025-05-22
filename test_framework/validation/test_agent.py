"""
Test agent that interprets test requirements and manages test execution.
This module bridges between browser-use's agent and our validation framework.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging
import re

from browser_use.browser.views import ActionResult
from browser_use.agent.views import AgentAction

from .assertions import TestAssertions, AssertionResult
from .validator import ValidationMode

logger = logging.getLogger("test_framework.validation.test_agent")

@dataclass
class TestRequirement:
    """A single test requirement to be verified"""
    description: str
    type: str  # text, link, attribute
    expected_value: Any
    selector: Optional[str] = None
    mode: ValidationMode = ValidationMode.EXACT
    metadata: Optional[Dict[str, Any]] = None

class TestAgent:
    """Interprets test requirements and manages test execution"""
    
    def __init__(self, browser_use_agent):
        """Initialize with browser-use's agent"""
        self.browser_use_agent = browser_use_agent
        self.assertions = TestAssertions()
    
    def _parse_requirement(self, requirement: str) -> TestRequirement:
        """Parse a natural language requirement into a structured test requirement"""
        # Extract requirement type
        if "link" in requirement.lower() and "href" in requirement.lower():
            # Link verification
            text_match = re.search(r"'([^']*)'", requirement)
            href_match = re.search(r'href=["\']([^"\']*)["\']', requirement)
            
            if text_match and href_match:
                return TestRequirement(
                    description=requirement,
                    type="link",
                    expected_value={
                        "text": text_match.group(1),
                        "href": href_match.group(1)
                    },
                    selector=text_match.group(1),
                    mode=ValidationMode.EXACT
                )
        elif "attribute" in requirement.lower():
            # Attribute verification
            selector_match = re.search(r'selector=["\']([^"\']*)["\']', requirement)
            attr_match = re.search(r'attribute=["\']([^"\']*)["\']', requirement)
            value_match = re.search(r'value=["\']([^"\']*)["\']', requirement)
            
            if selector_match and attr_match and value_match:
                return TestRequirement(
                    description=requirement,
                    type="attribute",
                    expected_value=value_match.group(1),
                    selector=selector_match.group(1),
                    metadata={"attribute": attr_match.group(1)},
                    mode=ValidationMode.EXACT
                )
        else:
            # Text verification
            text_match = re.search(r"'([^']*)'", requirement)
            if text_match:
                return TestRequirement(
                    description=requirement,
                    type="text",
                    expected_value=text_match.group(1),
                    selector=text_match.group(1),
                    mode=ValidationMode.CONTAINS
                )
        
        # If no specific type is detected, treat as a general condition
        return TestRequirement(
            description=requirement,
            type="text",  # Default to text verification
            expected_value=requirement,
            selector=requirement,
            mode=ValidationMode.CONTAINS
        )
    
    async def verify_requirement(self, requirement: str) -> AssertionResult:
        """Verify a single test requirement"""
        try:
            # Parse requirement
            test_req = self._parse_requirement(requirement)
            logger.info(f"Verifying requirement: {test_req.description}")
            
            # Get data from browser-use's agent
            action_result = None
            
            if test_req.type == "text":
                # Use browser-use's agent to get text
                action_result = await self.browser_use_agent.execute_action(
                    AgentAction(
                        type="get_text",
                        params={"text": test_req.selector}
                    )
                )
                
                # Validate text
                return self.assertions.verify_text(
                    action_result=action_result,
                    expected_text=test_req.expected_value,
                    exact=(test_req.mode == ValidationMode.EXACT),
                    original_requirement=test_req.description
                )
                
            elif test_req.type == "link":
                # Use browser-use's agent to get link
                action_result = await self.browser_use_agent.execute_action(
                    AgentAction(
                        type="get_link",
                        params={"text": test_req.expected_value["text"]}
                    )
                )
                
                # Validate link
                return self.assertions.verify_link(
                    action_result=action_result,
                    expected_text=test_req.expected_value["text"],
                    expected_href=test_req.expected_value["href"],
                    exact=(test_req.mode == ValidationMode.EXACT),
                    original_requirement=test_req.description
                )
                
            elif test_req.type == "attribute":
                # Use browser-use's agent to get attribute
                action_result = await self.browser_use_agent.execute_action(
                    AgentAction(
                        type="get_attribute",
                        params={
                            "selector": test_req.selector,
                            "attribute": test_req.metadata["attribute"]
                        }
                    )
                )
                
                # Validate attribute
                return self.assertions.verify_attribute(
                    action_result=action_result,
                    expected_value=test_req.expected_value,
                    original_requirement=test_req.description
                )
                
        except Exception as e:
            logger.error(f"Error verifying requirement: {str(e)}")
            return AssertionResult(
                success=False,
                message=f"Error verifying requirement: {str(e)}",
                error_code="UNKNOWN_ERROR",
                original_requirement=requirement
            )
    
    async def verify_requirements(self, requirements: List[str]) -> List[AssertionResult]:
        """Verify multiple test requirements"""
        results = []
        for requirement in requirements:
            result = await self.verify_requirement(requirement)
            results.append(result)
            if not result.success:
                logger.error(f"Requirement failed: {requirement}")
                logger.error(f"Error: {result.message}")
        return results 