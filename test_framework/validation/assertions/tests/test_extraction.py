import pytest
from test_framework.validation.assertions.extraction import ExtractionAssertions
from browser_use.agent.views import ActionResult, AgentHistoryList

class TestExtractionAssertions:
    @pytest.fixture
    def extraction_assertions(self):
        # Create a mock agent and result
        class MockAgent:
            def __init__(self):
                self.controller = None
                self.browser_session = None
                self.settings = None
                
        agent = MockAgent()
        result = AgentHistoryList(history=[])  # Initialize with empty history list
        return ExtractionAssertions(agent, result)

    def test_exact_match(self, extraction_assertions):
        # Test exact string match
        assert extraction_assertions._calculate_text_similarity("Hello", "Hello") == 1.0
        assert extraction_assertions._calculate_text_similarity("Hello World", "Hello World") == 1.0

    def test_contains_logic(self, extraction_assertions):
        # Test contains logic
        assert extraction_assertions._calculate_text_similarity("Hello", "Hello World") == 1.0
        assert extraction_assertions._calculate_text_similarity("World", "Hello World") == 1.0

    def test_json_exact_key_match(self, extraction_assertions):
        """Test exact match with JSON key."""
        json_content = {"Digital Content and Devices": "Some content"}
        assert extraction_assertions._calculate_text_similarity("Digital Content and Devices", json_content) == 1.0

    def test_json_contains_key(self, extraction_assertions):
        """Test contains match with JSON key."""
        json_content = {"Main Menu - Digital Content and Devices": "Some content"}
        assert extraction_assertions._calculate_text_similarity("Digital Content and Devices", json_content) == 1.0

    def test_json_exact_value_match(self, extraction_assertions):
        """Test exact match with JSON value."""
        json_content = {"Menu": "Digital Content and Devices"}
        assert extraction_assertions._calculate_text_similarity("Digital Content and Devices", json_content) == 1.0

    def test_json_contains_value(self, extraction_assertions):
        """Test contains match with JSON value."""
        json_content = {"Menu": "Digital Content and Devices section"}
        assert extraction_assertions._calculate_text_similarity("Digital Content and Devices", json_content) == 1.0

    def test_json_nested_exact_match(self, extraction_assertions):
        """Test exact match in nested JSON."""
        json_content = {
            "Menu": {
                "Digital Content and Devices": "Some content"
            }
        }
        assert extraction_assertions._calculate_text_similarity("Digital Content and Devices", json_content) == 1.0

    def test_json_nested_contains_match(self, extraction_assertions):
        """Test contains match in nested JSON."""
        json_content = {
            "Menu": {
                "Main - Digital Content and Devices": "Some content"
            }
        }
        assert extraction_assertions._calculate_text_similarity("Digital Content and Devices", json_content) == 1.0

    def test_json_similarity_fallback(self, extraction_assertions):
        """Test similarity score fallback for JSON."""
        json_content = {"Digital Content & Devices": "Some content"}
        similarity = extraction_assertions._calculate_text_similarity("Digital Content and Devices", json_content)
        assert 0.0 < similarity < 1.0  # Should be a partial match

    def test_json_no_match(self, extraction_assertions):
        """Test no match in JSON."""
        json_content = {"key": "World"}
        assert extraction_assertions._calculate_text_similarity("Hello", json_content) == 0.0

    def test_json_case_insensitive(self, extraction_assertions):
        """Test case-insensitive matching in JSON."""
        json_content = {"DIGITAL CONTENT AND DEVICES": "Some content"}
        assert extraction_assertions._calculate_text_similarity("Digital Content and Devices", json_content) == 1.0

    def test_json_whitespace_handling(self, extraction_assertions):
        """Test whitespace handling in JSON."""
        json_content = {"  Digital Content and Devices  ": "Some content"}
        assert extraction_assertions._calculate_text_similarity("Digital Content and Devices", json_content) == 1.0

    def test_no_match(self, extraction_assertions):
        # Test no match cases
        assert extraction_assertions._calculate_text_similarity("Hello", "World") == 0.0
        assert extraction_assertions._calculate_text_similarity("Hello", {"key": "World"}) == 0.0

    def test_case_insensitive(self, extraction_assertions):
        # Test case insensitivity
        assert extraction_assertions._calculate_text_similarity("HELLO", "hello") == 1.0
        assert extraction_assertions._calculate_text_similarity("HELLO", {"HELLO": "world"}) == 1.0

    def test_whitespace_handling(self, extraction_assertions):
        # Test whitespace handling
        assert extraction_assertions._calculate_text_similarity("  Hello  ", "Hello") == 1.0
        assert extraction_assertions._calculate_text_similarity("Hello", {"  Hello  ": "world"}) == 1.0

    def test_nested_json(self, extraction_assertions):
        # Test nested JSON structure
        json_content = {
            "Menu": {
                "Digital Content and Devices": "Some content"
            }
        }
        assert extraction_assertions._calculate_text_similarity("Digital Content and Devices", json_content) == 1.0

    def test_digital_content_exact_match(self, extraction_assertions):
        """Test exact matching for 'Digital Content and Devices'."""
        expected = "Digital Content and Devices"
        extracted = "Digital Content and Devices"
        assert extraction_assertions._calculate_text_similarity(expected, extracted) == 1.0
        
    def test_digital_content_with_context(self, extraction_assertions):
        """Test when extracted text contains additional context."""
        expected = "Digital Content and Devices"
        extracted = "Shop by Department\nDigital Content and Devices\nShop by Category"
        assert extraction_assertions._calculate_text_similarity(expected, extracted) == 1.0
        
    def test_digital_content_case_insensitive(self, extraction_assertions):
        """Test case-insensitive matching for 'Digital Content and Devices'."""
        expected = "Digital Content and Devices"
        extracted = "digital content and devices"
        assert extraction_assertions._calculate_text_similarity(expected, extracted) == 1.0
        
    def test_prime_music_exact_match(self, extraction_assertions):
        """Test exact matching for 'Amazon Prime Music'."""
        expected = "Amazon Prime Music"
        extracted = "Amazon Prime Music"
        assert extraction_assertions._calculate_text_similarity(expected, extracted) == 1.0
        
    def test_prime_music_with_context(self, extraction_assertions):
        """Test when extracted text contains additional context."""
        expected = "Amazon Prime Music"
        extracted = "Music, Movies & TV Shows\nAmazon Prime Music\nStreaming Services"
        assert extraction_assertions._calculate_text_similarity(expected, extracted) == 1.0
        
    def test_prime_music_typo(self, extraction_assertions):
        """Test when there's a typo in 'Amazon Prime Music'."""
        expected = "Amazon Prime Music"
        extracted = "Amazon Prime Msic"  # Intentional typo
        similarity = extraction_assertions._calculate_text_similarity(expected, extracted)
        assert similarity < 1.0  # Should use similarity score as fallback
        
    def test_digital_content_partial_match(self, extraction_assertions):
        """Test partial matching for 'Digital Content'."""
        expected = "Digital Content"
        extracted = "Digital Content and Devices"
        assert extraction_assertions._calculate_text_similarity(expected, extracted) == 1.0
        
    def test_prime_music_partial_match(self, extraction_assertions):
        """Test partial matching for 'Prime Music'."""
        expected = "Prime Music"
        extracted = "Amazon Prime Music"
        assert extraction_assertions._calculate_text_similarity(expected, extracted) == 1.0
        
    def test_digital_content_no_match(self, extraction_assertions):
        """Test when extracted text doesn't match 'Digital Content and Devices'."""
        expected = "Digital Content and Devices"
        extracted = "Amazon Prime Music"
        similarity = extraction_assertions._calculate_text_similarity(expected, extracted)
        assert similarity < 1.0  # Should use similarity score as fallback
        
    def test_prime_music_no_match(self, extraction_assertions):
        """Test when extracted text doesn't match 'Amazon Prime Music'."""
        expected = "Amazon Prime Music"
        extracted = "Digital Content and Devices"
        similarity = extraction_assertions._calculate_text_similarity(expected, extracted)
        assert similarity < 1.0  # Should use similarity score as fallback

    # New test cases for additional scenarios
    def test_nested_menu_structure(self, extraction_assertions):
        """Test matching within a nested menu structure."""
        expected = "Digital Content and Devices"
        extracted = """
        Shop by Department
            Electronics
                Digital Content and Devices
                    Prime Video
                    Prime Music
            Books
        """
        assert extraction_assertions._calculate_text_similarity(expected, extracted) == 1.0

    def test_multiple_occurrences(self, extraction_assertions):
        """Test when the text appears multiple times in the content."""
        expected = "Digital Content and Devices"
        extracted = """
        Digital Content and Devices
        Shop by Category
        Digital Content and Devices
        """
        assert extraction_assertions._calculate_text_similarity(expected, extracted) == 1.0

    def test_special_characters_in_context(self, extraction_assertions):
        """Test matching with special characters in the surrounding context."""
        expected = "Digital Content and Devices"
        extracted = "Shop by Department > Digital Content and Devices > Prime Video"
        assert extraction_assertions._calculate_text_similarity(expected, extracted) == 1.0

    def test_unicode_characters(self, extraction_assertions):
        """Test matching with Unicode characters in the text."""
        expected = "Digital Content and Devices"
        extracted = "Digital Content and Devices → Prime Video"
        assert extraction_assertions._calculate_text_similarity(expected, extracted) == 1.0

    def test_mixed_case_with_special_chars(self, extraction_assertions):
        """Test matching with mixed case and special characters."""
        expected = "Digital Content and Devices"
        extracted = "DIGITAL CONTENT & DEVICES"
        similarity = extraction_assertions._calculate_text_similarity(expected, extracted)
        assert similarity < 1.0  # Should use similarity score as fallback

    def test_whitespace_variations(self, extraction_assertions):
        """Test matching with various whitespace patterns."""
        expected = "Digital Content and Devices"
        extracted = "Digital\tContent\nand\r\nDevices"
        assert extraction_assertions._calculate_text_similarity(expected, extracted) == 1.0

    def test_html_tags(self, extraction_assertions):
        """Test matching when text is wrapped in HTML tags."""
        expected = "Digital Content and Devices"
        extracted = "<div>Digital Content and Devices</div>"
        assert extraction_assertions._calculate_text_similarity(expected, extracted) == 1.0

    def test_menu_with_numbers(self, extraction_assertions):
        """Test matching when text appears in a numbered menu."""
        expected = "Digital Content and Devices"
        extracted = "1. Shop by Department\n2. Digital Content and Devices\n3. Prime Video"
        assert extraction_assertions._calculate_text_similarity(expected, extracted) == 1.0

    # New test cases for JSON extraction format
    def test_digital_content_json_key(self, extraction_assertions):
        """Test when the text is a JSON key."""
        expected = "Digital Content and Devices"
        extracted = {
            "Digital Content and Devices": "* Echo & Alexa\n* Fire TV\n* Kindle E-Readers & eBooks\n* Audible Audiobooks\n* Amazon Prime Video\n* Amazon Prime Music"
        }
        assert extraction_assertions._calculate_text_similarity(expected, extracted) == 1.0

    def test_prime_music_json_key(self, extraction_assertions):
        """Test when the text is a JSON key with typo."""
        expected = "Amazon Prime Music"
        extracted = {
            "Amazon Prime Msic Page": "Streaming music service with 100 million songs"
        }
        similarity = extraction_assertions._calculate_text_similarity(expected, extracted)
        assert similarity < 1.0  # Should use similarity score as fallback

    def test_digital_content_json_value(self, extraction_assertions):
        """Test when the text is in a JSON value."""
        expected = "Digital Content and Devices"
        extracted = {
            "menu": "Digital Content and Devices\n* Echo & Alexa\n* Fire TV"
        }
        assert extraction_assertions._calculate_text_similarity(expected, extracted) == 1.0

    def test_prime_music_json_value(self, extraction_assertions):
        """Test when the text is in a JSON value with typo."""
        expected = "Amazon Prime Music"
        extracted = {
            "menu": "Amazon Prime Msic\n100 million songs, ad-free"
        }
        similarity = extraction_assertions._calculate_text_similarity(expected, extracted)
        assert similarity < 1.0  # Should use similarity score as fallback 