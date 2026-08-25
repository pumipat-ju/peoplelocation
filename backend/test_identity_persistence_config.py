import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class IdentityPersistenceConfigurationTests(unittest.TestCase):
    def test_compose_identity_database_uses_persistent_data_mount(self):
        compose = (REPOSITORY_ROOT / "docker-compose.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("- ./backend/data:/app/data", compose)
        self.assertIn(
            "- IDENTITY_DB_PATH=${IDENTITY_DB_PATH:-/app/data/identity_memory.sqlite3}",
            compose,
        )

    def test_example_environment_documents_container_database_path(self):
        example = (REPOSITORY_ROOT / ".env.example").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "IDENTITY_DB_PATH=/app/data/identity_memory.sqlite3",
            example,
        )


if __name__ == "__main__":
    unittest.main()
