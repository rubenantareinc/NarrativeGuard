from src.analysis.error_analyzer import ErrorAnalyzer


def test_error_analyzer_categorization():
    analyzer = ErrorAnalyzer()
    record = analyzer.categorize_error(
        input_text="Earlier he said the door was locked, later he said it was open.",
        expected="contradiction",
        predicted="consistent",
        context="Section 2",
    )
    assert record.category.startswith("false_negatives")
    summary = analyzer.generate_report()
    assert summary["total_errors"] == 1
