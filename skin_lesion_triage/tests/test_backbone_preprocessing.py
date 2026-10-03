import cv2
import numpy as np
import pytest
import tensorflow as tf

from app import preprocessing
from app.backbone_preprocessing import preprocess_rgb_for_model
from app.image_preprocessing import preprocess_bgr_image
from scripts.train_model import _load_directory, _prepare_dataset, build_augmentation


@pytest.mark.parametrize("architecture", ["mobilenetv2", "resnet50"])
def test_training_and_inference_backbone_preprocessing_have_parity(architecture):
    rgb_pixels = tf.constant(
        [[[[10.0, 20.0, 30.0], [255.0, 128.0, 0.0]]]],
        dtype=tf.float32,
    )
    labels = tf.constant([[0.0]])
    train_ds = tf.data.Dataset.from_tensor_slices((rgb_pixels, labels)).batch(1)
    trained_pixels = next(iter(_prepare_dataset(train_ds, architecture)))[0].numpy()
    inference_pixels = preprocess_rgb_for_model(rgb_pixels, architecture).numpy()

    np.testing.assert_allclose(trained_pixels, inference_pixels, atol=1e-5)
    assert trained_pixels.shape == (1, 1, 2, 3)
    assert trained_pixels.dtype == np.float32


def test_architecture_preprocessing_preserves_documented_imagenet_semantics():
    rgb = tf.constant([[[[10.0, 20.0, 30.0]]]])
    mobile = preprocess_rgb_for_model(rgb, "mobilenetv2").numpy()[0, 0, 0]
    resnet = preprocess_rgb_for_model(rgb, "resnet50").numpy()[0, 0, 0]

    np.testing.assert_allclose(mobile, np.array([10, 20, 30]) / 127.5 - 1.0, atol=1e-6)
    np.testing.assert_allclose(
        resnet,
        np.array([30 - 103.939, 20 - 116.779, 10 - 123.68]),
        atol=1e-3,
    )


def test_preprocessor_rejects_invalid_architecture_shape_channels_and_range():
    with pytest.raises(ValueError, match="Unsupported architecture"):
        preprocess_rgb_for_model(tf.zeros((1, 2, 2, 3)), "unknown")
    with pytest.raises(ValueError, match="rank 4"):
        preprocess_rgb_for_model(tf.zeros((2, 2, 3)), "mobilenetv2")
    with pytest.raises(ValueError, match="three RGB channels"):
        preprocess_rgb_for_model(tf.zeros((1, 2, 2, 1)), "mobilenetv2")
    with pytest.raises(tf.errors.InvalidArgumentError, match="at most 255"):
        preprocess_rgb_for_model(tf.fill((1, 2, 2, 3), 256.0), "mobilenetv2")


@pytest.mark.parametrize("architecture", ["mobilenetv2", "resnet50"])
def test_image_bytes_return_batched_architecture_specific_tensor(monkeypatch, architecture):
    monkeypatch.setattr(preprocessing, "enhance_bgr_image", lambda image: image)
    monkeypatch.setattr(preprocessing, "check_quality", lambda image: 120.0)
    bgr = np.zeros((16, 16, 3), dtype=np.uint8)
    bgr[:] = (30, 20, 10)
    success, encoded = cv2.imencode(".png", bgr)
    assert success

    result, sharpness = preprocessing.preprocess_image(
        encoded.tobytes(),
        target_size=(16, 16),
        architecture=architecture,
    )

    expected = preprocess_rgb_for_model(
        tf.constant(np.full((1, 16, 16, 3), (10, 20, 30), dtype=np.float32)),
        architecture,
    ).numpy()
    assert result.shape == (1, 16, 16, 3)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, expected, atol=1e-5)
    assert sharpness == 120.0


def test_training_and_serving_share_full_opencv_transform(tmp_path, monkeypatch):
    rng = np.random.default_rng(123)
    bgr = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    image_path = tmp_path / "train" / "benign" / "same.jpg"
    image_path.parent.mkdir(parents=True)
    malignant_dir = tmp_path / "train" / "malignant"
    malignant_dir.mkdir()
    assert cv2.imwrite(str(image_path), bgr)
    assert cv2.imwrite(str(malignant_dir / "other.jpg"), bgr)
    image_bytes = image_path.read_bytes()
    monkeypatch.setattr(preprocessing, "check_quality", lambda image: 120.0)

    train_raw = _load_directory(tmp_path / "train")
    training_images, _ = next(iter(_prepare_dataset(train_raw, "mobilenetv2")))
    serving_images, _ = preprocessing.preprocess_image(
        image_bytes,
        target_size=(224, 224),
        architecture="mobilenetv2",
    )

    np.testing.assert_allclose(training_images[:1].numpy(), serving_images, atol=1e-5)
    expected_rgb = preprocess_bgr_image(cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR))
    assert expected_rgb.shape == (224, 224, 3)
    assert expected_rgb.dtype == np.uint8
    expected_model_input = preprocess_rgb_for_model(
        tf.convert_to_tensor(expected_rgb[None, ...], dtype=tf.float32),
        "mobilenetv2",
    ).numpy()
    np.testing.assert_allclose(serving_images, expected_model_input, atol=1e-5)
    assert training_images.shape == (2, 224, 224, 3)


def test_brightness_augmentation_range_matches_normalized_input():
    augmentation = build_augmentation()
    brightness = next(layer for layer in augmentation.layers if isinstance(layer, tf.keras.layers.RandomBrightness))
    assert list(brightness.value_range) == [0.0, 1.0]
