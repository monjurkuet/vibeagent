"""Multi-modal Skill for handling images, videos, and code.

This skill processes and ingests multi-modal content (images, videos, code)
into the knowledge base with appropriate embeddings and metadata.
"""

import base64
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.skill import BaseSkill, SkillResult

logger = logging.getLogger(__name__)


@dataclass
class MultiModalDocument:
    """Multi-modal document."""

    id: str
    content_type: str  # "image", "video", "code", "text"
    content: str | bytes
    embedding: list[float] | None = None
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class MultiModalSkill(BaseSkill):
    """Skill for processing multi-modal content."""

    def __init__(
        self,
        embedding_model: str | None = None,
        max_image_size: int = 10 * 1024 * 1024,  # 10MB
        max_video_size: int = 100 * 1024 * 1024,  # 100MB
        vision_model: str = "qwen3-vl-plus",
        vision_api_url: str = "http://localhost:8087/v1",
    ):
        """Initialize Multi-modal skill.

        Args:
            embedding_model: Model name for embeddings
            max_image_size: Maximum image size in bytes
            max_video_size: Maximum video size in bytes
            vision_model: Vision model for image analysis (qwen3-vl-plus)
            vision_api_url: Base URL for vision model API
        """
        super().__init__(
            name="multimodal",
            version="1.0.0",
        )
        self.embedding_model = embedding_model
        self.max_image_size = max_image_size
        self.max_video_size = max_video_size
        self.vision_model = vision_model
        self.vision_api_url = vision_api_url.rstrip("/")
        self.vision_chat_url = f"{self.vision_api_url}/chat/completions"
        self._image_embedding_model = None
        self._text_embedding_model = None

    def _get_image_embedding_model(self) -> Any:
        """Get or create image embedding model."""
        if self._image_embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                model_name = self.embedding_model or "clip-ViT-B-32"
                self._image_embedding_model = SentenceTransformer(model_name)
            except ImportError:
                raise ImportError(
                    "Sentence transformers not installed. Install with: pip install sentence-transformers"
                )
        return self._image_embedding_model

    def _get_text_embedding_model(self) -> Any:
        """Get or create text embedding model."""
        if self._text_embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                model_name = self.embedding_model or "all-MiniLM-L6-v2"
                self._text_embedding_model = SentenceTransformer(model_name)
            except ImportError:
                raise ImportError(
                    "Sentence transformers not installed. Install with: pip install sentence-transformers"
                )
        return self._text_embedding_model

    def validate(self) -> bool:
        """Validate skill configuration."""
        return True

    def get_dependencies(self) -> list[str]:
        """Return list of dependencies."""
        deps = ["pillow", "python-magic", "requests"]
        if self.embedding_model:
            deps.append("sentence-transformers")
        return deps

    def get_tool_schema(self) -> dict[str, Any]:
        """Get OpenAI function schema for this skill."""
        return {
            "type": "function",
            "function": {
                "name": "process_multimodal",
                "description": "Process multi-modal content (images, videos, code)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content_type": {
                            "type": "string",
                            "enum": ["image", "video", "code", "text"],
                            "description": "Type of content to process",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content (file path, base64, or text)",
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Additional metadata",
                        },
                    },
                    "required": ["content_type", "content"],
                },
            },
        }

    def execute(
        self,
        action: str = "process",
        content_type: str | None = None,
        content: str | None = None,
        file_path: str | None = None,
        metadata: dict | None = None,
        **kwargs,
    ) -> SkillResult:
        """Execute Multi-modal skill.

        Args:
            action: Action to perform (process, extract_text, generate_caption, analyze)
            content_type: Type of content (image, video, code, text)
            content: Content data
            file_path: Path to file
            metadata: Additional metadata

        Returns:
            SkillResult with processing results
        """
        if action == "process":
            return self._process_content(content_type, content, file_path, metadata, **kwargs)
        elif action == "extract_text":
            return self._extract_text_from_image(content, file_path, **kwargs)
        elif action == "generate_caption":
            return self._generate_caption(content, file_path, **kwargs)
        elif action == "analyze":
            return self._analyze_image(content, file_path, **kwargs)
        elif action == "process_code":
            return self._process_code(content, file_path, metadata, **kwargs)
        else:
            return SkillResult(success=False, error=f"Unknown action: {action}")

    def _process_content(
        self,
        content_type: str,
        content: str | None,
        file_path: str | None,
        metadata: dict | None,
        **kwargs,
    ) -> SkillResult:
        """Process multi-modal content.

        Args:
            content_type: Type of content
            content: Content data
            file_path: Path to file
            metadata: Additional metadata

        Returns:
            SkillResult with processed content
        """
        try:
            if content_type == "image":
                return self._process_image(content, file_path, metadata, **kwargs)
            elif content_type == "video":
                return self._process_video(file_path, metadata, **kwargs)
            elif content_type == "code":
                return self._process_code(content, file_path, metadata, **kwargs)
            elif content_type == "text":
                return self._process_text(content, metadata, **kwargs)
            else:
                return SkillResult(success=False, error=f"Unknown content type: {content_type}")

        except Exception as e:
            self._record_error()
            logger.error(f"Error processing content: {e}")
            return SkillResult(success=False, error=f"Processing failed: {str(e)}")

    def _process_image(
        self,
        content: str | None,
        file_path: str | None,
        metadata: dict | None,
        **kwargs,
    ) -> SkillResult:
        """Process image content.

        Args:
            content: Image data (base64 or path)
            file_path: Path to image file
            metadata: Additional metadata

        Returns:
            SkillResult with processed image
        """
        try:
            from PIL import Image

            # Load image
            if file_path:
                image = Image.open(file_path)
            elif content:
                # Assume base64 encoded
                image_data = base64.b64decode(content)
                image = Image.open(io.BytesIO(image_data))
            else:
                return SkillResult(success=False, error="No image content provided")

            # Check file size
            if file_path and os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                if file_size > self.max_image_size:
                    return SkillResult(
                        success=False, error=f"Image too large: {file_size} bytes (max: {self.max_image_size})"
                    )

            # Generate embedding
            model = self._get_image_embedding_model()
            embedding = model.encode(image, convert_to_numpy=False).tolist()

            # Extract image info
            image_info = {
                "format": image.format,
                "mode": image.mode,
                "size": image.size,
                "width": image.width,
                "height": image.height,
            }

            # Create document
            doc_id = f"image_{hash(embedding)}"
            doc = MultiModalDocument(
                id=doc_id,
                content_type="image",
                content=content or file_path or "",
                embedding=embedding,
                metadata={
                    **image_info,
                    **(metadata or {}),
                },
            )

            self._record_usage()
            return SkillResult(
                success=True,
                data={
                    "id": doc.id,
                    "content_type": doc.content_type,
                    "embedding_size": len(embedding),
                    "metadata": doc.metadata,
                },
            )

        except ImportError:
            return SkillResult(success=False, error="PIL not installed. Install with: pip install pillow")
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return SkillResult(success=False, error=f"Image processing failed: {str(e)}")

    def _process_video(
        self,
        file_path: str | None,
        metadata: dict | None,
        **kwargs,
    ) -> SkillResult:
        """Process video content.

        Args:
            file_path: Path to video file
            metadata: Additional metadata

        Returns:
            SkillResult with processed video
        """
        try:
            if not file_path:
                return SkillResult(success=False, error="Video file path required")

            # Check file size
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                if file_size > self.max_video_size:
                    return SkillResult(
                        success=False, error=f"Video too large: {file_size} bytes (max: {self.max_video_size})"
                    )

            # Get video metadata using python-magic
            import magic

            mime = magic.Magic(mime=True)
            file_type = mime.from_file(file_path)

            # For videos, we typically need to extract frames
            # This is a simplified version
            doc_id = f"video_{hash(file_path)}"
            doc = MultiModalDocument(
                id=doc_id,
                content_type="video",
                content=file_path,
                metadata={
                    "file_type": file_type,
                    "file_size": file_size,
                    **(metadata or {}),
                },
            )

            self._record_usage()
            return SkillResult(
                success=True,
                data={
                    "id": doc.id,
                    "content_type": doc.content_type,
                    "metadata": doc.metadata,
                    "note": "Video embedding requires frame extraction",
                },
            )

        except ImportError:
            return SkillResult(success=False, error="python-magic not installed. Install with: pip install python-magic")
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return SkillResult(success=False, error=f"Video processing failed: {str(e)}")

    def _process_code(
        self,
        content: str | None,
        file_path: str | None,
        metadata: dict | None,
        **kwargs,
    ) -> SkillResult:
        """Process code content.

        Args:
            content: Code content
            file_path: Path to code file
            metadata: Additional metadata

        Returns:
            SkillResult with processed code
        """
        try:
            # Load code content
            if file_path:
                with open(file_path) as f:
                    code_content = f.read()
            elif content:
                code_content = content
            else:
                return SkillResult(success=False, error="No code content provided")

            # Detect language
            language = self._detect_code_language(file_path, code_content)

            # Generate embedding for code
            model = self._get_text_embedding_model()
            embedding = model.encode(code_content, convert_to_numpy=False).tolist()

            # Create document
            doc_id = f"code_{hash(code_content)}"
            doc = MultiModalDocument(
                id=doc_id,
                content_type="code",
                content=code_content,
                embedding=embedding,
                metadata={
                    "language": language,
                    "file_path": file_path,
                    "line_count": len(code_content.splitlines()),
                    **(metadata or {}),
                },
            )

            self._record_usage()
            return SkillResult(
                success=True,
                data={
                    "id": doc.id,
                    "content_type": doc.content_type,
                    "language": language,
                    "line_count": doc.metadata.get("line_count"),
                    "embedding_size": len(embedding),
                    "metadata": doc.metadata,
                },
            )

        except Exception as e:
            logger.error(f"Error processing code: {e}")
            return SkillResult(success=False, error=f"Code processing failed: {str(e)}")

    def _process_text(
        self,
        content: str | None,
        metadata: dict | None,
        **kwargs,
    ) -> SkillResult:
        """Process text content.

        Args:
            content: Text content
            metadata: Additional metadata

        Returns:
            SkillResult with processed text
        """
        try:
            if not content:
                return SkillResult(success=False, error="No text content provided")

            # Generate embedding
            model = self._get_text_embedding_model()
            embedding = model.encode(content, convert_to_numpy=False).tolist()

            # Create document
            doc_id = f"text_{hash(content)}"
            doc = MultiModalDocument(
                id=doc_id,
                content_type="text",
                content=content,
                embedding=embedding,
                metadata={
                    "char_count": len(content),
                    "word_count": len(content.split()),
                    **(metadata or {}),
                },
            )

            self._record_usage()
            return SkillResult(
                success=True,
                data={
                    "id": doc.id,
                    "content_type": doc.content_type,
                    "char_count": doc.metadata.get("char_count"),
                    "word_count": doc.metadata.get("word_count"),
                    "embedding_size": len(embedding),
                    "metadata": doc.metadata,
                },
            )

        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return SkillResult(success=False, error=f"Text processing failed: {str(e)}")

    def _extract_text_from_image(
        self,
        content: str | None,
        file_path: str | None,
        **kwargs,
    ) -> SkillResult:
        """Extract text from image using OCR.

        Args:
            content: Image data
            file_path: Path to image file

        Returns:
            SkillResult with extracted text
        """
        try:
            # This would require OCR libraries like pytesseract
            # For now, return a placeholder
            return SkillResult(
                success=False,
                error="OCR extraction requires pytesseract. Install with: pip install pytesseract",
            )

        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return SkillResult(success=False, error=f"Text extraction failed: {str(e)}")

    def _generate_caption(
        self,
        content: str | None,
        file_path: str | None,
        **kwargs,
    ) -> SkillResult:
        """Generate caption for image using qwen3-vl-plus vision model.

        Args:
            content: Image data (base64 or path)
            file_path: Path to image file

        Returns:
            SkillResult with caption
        """
        try:
            import requests

            # Load and encode image
            if file_path:
                with open(file_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode("utf-8")
            elif content:
                # Assume content is already base64 encoded
                if not content.startswith("data:image"):
                    image_data = content
                else:
                    # Extract base64 part from data URL
                    image_data = content.split(",")[1]
            else:
                return SkillResult(success=False, error="No image content provided")

            # Construct message with image for vision model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                        {"type": "text", "text": "Describe this image in detail. What do you see?"},
                    ],
                }
            ]

            # Call vision model via OpenAI-compatible API
            response = requests.post(
                self.vision_chat_url,
                json={
                    "model": self.vision_model,
                    "messages": messages,
                    "max_tokens": 500,
                },
                timeout=30,
            )
            response.raise_for_status()

            result = response.json()
            caption = result["choices"][0]["message"]["content"]

            logger.info(f"Generated caption using {self.vision_model}")

            return SkillResult(
                success=True,
                data={
                    "caption": caption,
                    "model": self.vision_model,
                },
            )

        except requests.exceptions.HTTPError as e:
            error_detail = e.response.json() if e.response else str(e)
            logger.error(f"Vision model HTTP error: {error_detail}")
            return SkillResult(success=False, error=f"Vision model error: {error_detail}")
        except requests.exceptions.Timeout:
            logger.error("Vision model request timed out")
            return SkillResult(success=False, error="Vision model request timed out")
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return SkillResult(success=False, error=f"Caption generation failed: {str(e)}")

    def _analyze_image(
        self,
        content: str | None,
        file_path: str | None,
        analysis_type: str = "general",
        **kwargs,
    ) -> SkillResult:
        """Analyze image using qwen3-vl-plus vision model.

        Args:
            content: Image data (base64 or path)
            file_path: Path to image file
            analysis_type: Type of analysis (general, ocr, objects, scene)

        Returns:
            SkillResult with analysis results
        """
        try:
            import requests

            # Load and encode image
            if file_path:
                with open(file_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode("utf-8")
            elif content:
                # Assume content is already base64 encoded
                if not content.startswith("data:image"):
                    image_data = content
                else:
                    # Extract base64 part from data URL
                    image_data = content.split(",")[1]
            else:
                return SkillResult(success=False, error="No image content provided")

            # Build prompt based on analysis type
            prompts = {
                "general": "Analyze this image in detail. Describe the main elements, style, mood, and any notable features.",
                "ocr": "Extract all text visible in this image. Provide the exact text content.",
                "objects": "Identify and list all objects visible in this image. Be specific.",
                "scene": "Describe the scene in this image. What is happening? What is the setting?",
            }

            prompt = prompts.get(analysis_type, prompts["general"])

            # Construct message with image for vision model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Call vision model via OpenAI-compatible API
            response = requests.post(
                self.vision_chat_url,
                json={
                    "model": self.vision_model,
                    "messages": messages,
                    "max_tokens": 1000,
                },
                timeout=30,
            )
            response.raise_for_status()

            result = response.json()
            analysis = result["choices"][0]["message"]["content"]

            logger.info(f"Analyzed image using {self.vision_model} (type: {analysis_type})")

            return SkillResult(
                success=True,
                data={
                    "analysis": analysis,
                    "analysis_type": analysis_type,
                    "model": self.vision_model,
                },
            )

        except requests.exceptions.HTTPError as e:
            error_detail = e.response.json() if e.response else str(e)
            logger.error(f"Vision model HTTP error: {error_detail}")
            return SkillResult(success=False, error=f"Vision model error: {error_detail}")
        except requests.exceptions.Timeout:
            logger.error("Vision model request timed out")
            return SkillResult(success=False, error="Vision model request timed out")
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return SkillResult(success=False, error=f"Image analysis failed: {str(e)}")

    def _detect_code_language(self, file_path: str | None, code: str) -> str:
        """Detect programming language.

        Args:
            file_path: File path
            code: Code content

        Returns:
            Language name
        """
        if file_path:
            ext = Path(file_path).suffix.lower()
            language_map = {
                ".py": "python",
                ".js": "javascript",
                ".ts": "typescript",
                ".java": "java",
                ".cpp": "c++",
                ".c": "c",
                ".go": "go",
                ".rs": "rust",
                ".rb": "ruby",
                ".php": "php",
                ".sh": "bash",
                ".sql": "sql",
                ".html": "html",
                ".css": "css",
                ".json": "json",
                ".yaml": "yaml",
                ".yml": "yaml",
                ".xml": "xml",
                ".md": "markdown",
            }
            return language_map.get(ext, "unknown")

        return "unknown"

    def health_check(self) -> bool:
        """Check if skill is operational."""
        return True


# Import io for BytesIO
import io