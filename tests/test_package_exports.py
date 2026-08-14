def test_public_package_exports():
    import verification

    for name in (
        "Verdict",
        "VerificationResult",
        "VerificationAgent",
        "ClaimExtractor",
        "EvidenceRetriever",
        "VerdictAgent",
    ):
        assert hasattr(verification, name), f"missing export: {name}"
