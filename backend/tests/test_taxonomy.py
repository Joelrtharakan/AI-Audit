from app.services.taxonomy import RootCauseCategory, coerce_category


def test_valid_category_passes_through():
    assert coerce_category("METHOD") == RootCauseCategory.METHOD


def test_lowercase_and_spacing_normalized():
    assert coerce_category("environment mgmt") == RootCauseCategory.ENVIRONMENT_MGMT


def test_unknown_category_falls_back_to_other():
    assert coerce_category("SOMETHING_MADE_UP") == RootCauseCategory.OTHER


def test_empty_falls_back_to_other():
    assert coerce_category(None) == RootCauseCategory.OTHER
    assert coerce_category("") == RootCauseCategory.OTHER
