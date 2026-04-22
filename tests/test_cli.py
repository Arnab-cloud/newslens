from typer.testing import CliRunner

from newslens.cli.main import app

runner = CliRunner()


# def test_cli_train_command():
#     # We test that the train command accepts arguments correctly
#     # without actually needing to run a 3-hour training loop.
#     result = runner.invoke(
#         app, ["train", "fake_train.csv", "fake_val.csv", "--output-dir", "./test_out"]
#     )

#     # We expect an error because the files don't exist,
#     # but it confirms the CLI parsing logic is working.
#     assert result.exit_code != 0
#     assert "fake_train.csv" in str(result.output)


def test_cli_serve_help():
    # Verify the CLI help menu is accessible
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    # assert "Start the NewsLens inference service" in result.stdout
