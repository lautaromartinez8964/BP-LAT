"""Self-contained smoke test of raster input, prediction and NoData output."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin


class CommandLineTests(unittest.TestCase):
    def test_geotiff_round_trip(self):
        rng = np.random.default_rng(3)
        values = rng.normal(4, .35, (120, 120))
        values[20:95, 25:80] = rng.normal(1, .2, (75, 55))
        values[5:15, :] = rng.normal(7, .25, (10, 120))
        values[0, 0] = -9999
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path, output_dir = root / "input.tif", root / "output"
            transform = from_origin(400000, 2000000, 10, 10)
            with rasterio.open(input_path, "w", driver="GTiff", height=120, width=120, count=1,
                               dtype="float32", nodata=-9999, crs="EPSG:32648", transform=transform) as ds:
                ds.write(values.astype("float32"), 1)
            command = [sys.executable, "-m", "bplat.cli", "--input", str(input_path),
                       "--output-dir", str(output_dir), "--representation", "positive"]
            subprocess.run(command, check=True, env=os.environ.copy(), capture_output=True, text=True)
            with rasterio.open(output_dir / "mask.tif") as ds:
                data = ds.read(1)
                self.assertEqual(ds.transform, transform)
                self.assertEqual(ds.nodata, 255)
                self.assertEqual(data[0, 0], 255)
                self.assertTrue(np.isin(data, [0, 1, 255]).all())
                self.assertGreater(np.count_nonzero(data == 1), 0)
            meta = json.loads((output_dir / "run.json").read_text(encoding="utf8"))
            self.assertEqual(len(meta["input_sha256"]), 64)
            self.assertGreater(meta["accepted_sources"], 0)
            self.assertIn("numpy", meta["software_versions"])


if __name__ == "__main__":
    unittest.main()
