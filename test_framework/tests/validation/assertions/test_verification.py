@pytest.mark.asyncio
async def test_verify_condition_enhanced(verification_assertions, mock_result):
    """Test enhanced verification with fuzzy matching and substring matching."""
    # Setup mock result with extractions
    mock_action = MagicMock()
    mock_action.extract_content = {
        "text": "This is a sample text with some content to match against",
        "elements": [
            {"text": "First element"},
            {"text": "Second element"}
        ]
    }
    mock_result.append(mock_action)
    
    # Test exact match
    result = await verification_assertions._verify_condition(
        requirement='verify text "This is a sample text"',
        current_step=0
    )
    assert result.success
    assert result.metadata["match_results"][0]["match_type"] == "exact"
    assert result.metadata["match_results"][0]["matched_snippet"] == "This is a sample text"
    
    # Test case-insensitive match
    result = await verification_assertions._verify_condition(
        requirement='verify text "THIS IS A SAMPLE TEXT"',
        current_step=0
    )
    assert result.success
    assert result.metadata["match_results"][0]["match_type"] == "case_insensitive"
    
    # Test fuzzy match with substring
    result = await verification_assertions._verify_condition(
        requirement='verify text "sample txt"',
        current_step=0
    )
    assert result.success
    assert result.metadata["match_results"][0]["match_type"] == "fuzzy"
    assert result.metadata["match_results"][0]["similarity_score"] >= 0.85
    
    # Test fuzzy matching disabled
    result = await verification_assertions._verify_condition(
        requirement='verify text "sample txt"',
        current_step=0,
        use_fuzzy_matching=False
    )
    assert not result.success
    assert result.metadata["match_results"][0]["match_type"] == "no_match"
    
    # Test contains match
    result = await verification_assertions._verify_condition(
        requirement='verify text "content to match"',
        current_step=0
    )
    assert result.success
    assert result.metadata["match_results"][0]["match_type"] == "contains"
    
    # Test multiple quoted texts
    result = await verification_assertions._verify_condition(
        requirement='verify text "First element" and "Second element"',
        current_step=0
    )
    assert result.success
    assert len(result.metadata["match_results"]) == 2
    assert all(r["success"] for r in result.metadata["match_results"])
    
    # Test with nested structure
    mock_action.extract_content = {
        "menu": {
            "items": [
                {"name": "Home", "text": "Welcome"},
                {"name": "About", "text": "About Us"}
            ]
        }
    }
    result = await verification_assertions._verify_condition(
        requirement='verify text "Welcome" and "About Us"',
        current_step=0
    )
    assert result.success
    assert len(result.metadata["match_results"]) == 2
    assert all(r["success"] for r in result.metadata["match_results"])
    
    # Test with fuzzy match threshold
    mock_action.extract_content = {"text": "This is a slightly different text"}
    result = await verification_assertions._verify_condition(
        requirement='verify text "This is a slightly diffrent text"',
        current_step=0
    )
    assert result.success
    assert result.metadata["match_results"][0]["match_type"] == "fuzzy"
    assert result.metadata["match_results"][0]["similarity_score"] >= 0.85
    
    # Test with no match
    result = await verification_assertions._verify_condition(
        requirement='verify text "Completely different text"',
        current_step=0
    )
    assert not result.success
    assert result.metadata["match_results"][0]["match_type"] == "no_match" 