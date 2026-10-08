import torch

from streamnav.models.vision.preprocessing import patchify


def test_patchify_preserves_rgb_temporal_repeat_and_merge_order():
    image = torch.arange(64 * 64 * 3).remainder(256).to(torch.uint8).reshape(64, 64, 3)
    patches, grid = patchify(image, 64)
    assert grid.tolist() == [[1, 4, 4]]
    decoded = patches.view(16, 3, 2, 16, 16)
    torch.testing.assert_close(decoded[:, :, 0], decoded[:, :, 1])
    coords = [(0, 0), (0, 16), (16, 0), (16, 16), (0, 32)]
    for index, (y, x) in enumerate(coords):
        expected = image[y : y + 16, x : x + 16].permute(2, 0, 1).float() / 127.5 - 1
        torch.testing.assert_close(decoded[index, :, 0], expected)


def test_widescreen_preserves_every_sensor_pixel_and_pads_only():
    rgb = torch.arange(270 * 480 * 3).remainder(256).to(torch.uint8).reshape(270, 480, 3)
    patches, grid = patchify(rgb, [270, 480])
    assert grid.tolist() == [[1, 18, 30]]
    decoded = patches.view(1, 9, 15, 2, 2, 3, 2, 16, 16)
    decoded = decoded.permute(0, 5, 6, 1, 3, 7, 2, 4, 8).reshape(1, 3, 2, 288, 480)
    expected = rgb.permute(2, 0, 1).float() / 127.5 - 1
    torch.testing.assert_close(decoded[0, :, 0, 9:279], expected, rtol=0, atol=0)
    torch.testing.assert_close(decoded[:, :, 0], decoded[:, :, 1], rtol=0, atol=0)
    assert not decoded[..., :9, :].any()
    assert not decoded[..., 279:, :].any()
