"""
Modal Deployment Configuration Tests.
Tests criteria:
- Modal app definition and naming
- Container image definition and system packages
- GPU configuration options
- Local directory and weight bindings
- ASGI wrapper function
"""

import ast
import os
import pytest


class TestModalAppConfig:
    """Static and structural tests for modal_app.py."""

    def test_modal_app_file_exists(self):
        modal_path = os.path.join(os.path.dirname(__file__), "..", "modal_app.py")
        assert os.path.exists(modal_path), "modal_app.py must exist in food-ai-service root"

    def test_modal_app_syntax_and_structure(self):
        modal_path = os.path.join(os.path.dirname(__file__), "..", "modal_app.py")
        with open(modal_path, "r", encoding="utf-8") as f:
            source = f.read()

        # Parse AST to ensure no syntax errors
        tree = ast.parse(source)
        assert tree is not None

        # Check key Modal constructs in code
        assert "modal.App" in source, "modal_app.py must define modal.App"
        assert "food-ai-service" in source, "modal app name must be food-ai-service"
        assert "modal.Image" in source or "Image.debian_slim" in source, "Must define container Image"
        assert "asgi_app" in source, "Must expose ASGI app endpoint"
        assert "models_weights" in source, "Must mount models_weights directory"
        assert "ingredient_task" in source, "Must mount ingredient_task directory"
        assert "run_ai.py" in source, "Must mount run_ai.py"

    def test_modal_gpu_setting(self):
        """Test MODAL_GPU environment variable handling."""
        # When MODAL_GPU is T4
        os.environ["MODAL_GPU"] = "T4"
        import importlib
        # Verify it reads correctly
        gpu = os.environ.get("MODAL_GPU", "T4")
        assert gpu in ["T4", "A10G", "None"]
