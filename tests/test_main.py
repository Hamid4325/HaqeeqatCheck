from ingestion import main as cli


class TestMain:
    def test_prints_report_json(self, tmp_media, monkeypatch, capsys):
        class FakeIngestor:
            def ingest(self, path):
                return {
                    "file_type": "image",
                    "audio_transcript": "",
                    "ocr_text": "hi",
                    "combined_text": "[SCREEN TEXT]: hi",
                    "metadata": {"warnings": []},
                }

        monkeypatch.setattr(cli, "HaqeeqatIngestor", lambda: FakeIngestor())
        rc = cli.main([tmp_media.image()])
        out = capsys.readouterr().out
        assert rc == 0
        assert '"file_type": "image"' in out

    def test_unsupported_file_returns_2(self, tmp_path, capsys):
        bogus = tmp_path / "notes.txt"
        bogus.write_text("hi", encoding="utf-8")
        rc = cli.main([str(bogus)])
        err = capsys.readouterr().err
        assert rc == 2
        assert "Error:" in err


def test_package_exports():
    import ingestion

    assert ingestion.HaqeeqatIngestor is not None
    assert ingestion.WhisperTranscriber is not None
    assert ingestion.PaddleOCREngine is not None
    assert ingestion.Transcriber is not None
    assert ingestion.OCREngine is not None
    assert ingestion.UnsupportedFormatError is not None
