"""
Point cloud to image conversion classes
"""
import numpy as np


class BEVConverter:
    """Convert 4D radar point cloud to Bird's Eye View (BEV) image"""
    
    def __init__(self, x_range=(-50, 50), y_range=(-50, 50), 
                 resolution=0.2, height_range=(-2, 5)):
        """
        Args:
            x_range: (min, max) in meters for X axis (forward)
            y_range: (min, max) in meters for Y axis (left-right)
            resolution: grid cell size in meters
            height_range: (min, max) in meters for Z filtering
        """
        self.x_range = x_range
        self.y_range = y_range
        self.resolution = resolution
        self.height_range = height_range
        
        self.x_size = int((x_range[1] - x_range[0]) / resolution)
        self.y_size = int((y_range[1] - y_range[0]) / resolution)
        
    def point_to_pixel(self, points):
        """Convert XY coordinates to pixel indices"""
        x_img = ((points[:, 0] - self.x_range[0]) / self.resolution).astype(np.int32)
        y_img = ((points[:, 1] - self.y_range[0]) / self.resolution).astype(np.int32)
        return x_img, y_img
    
    def __call__(self, pc):
        """
        Convert point cloud to BEV image
        
        Args:
            pc: (N, 5) array [x, y, z, doppler, intensity]
            
        Returns:
            bev_image: (H, W, 4) array with channels:
                0: point density (log scale)
                1: mean intensity
                2: max height
                3: mean doppler velocity
        """
        # Filter by height range
        if self.height_range:
            mask = (pc[:, 2] >= self.height_range[0]) & (pc[:, 2] <= self.height_range[1])
            pc = pc[mask]
        
        if len(pc) == 0:
            return np.zeros((self.x_size, self.y_size, 4), dtype=np.float32)
        
        # Convert to pixel coordinates
        x_img, y_img = self.point_to_pixel(pc)
        
        # Filter out-of-range points
        mask = (x_img >= 0) & (x_img < self.x_size) & \
               (y_img >= 0) & (y_img < self.y_size)
        x_img = x_img[mask]
        y_img = y_img[mask]
        pc_valid = pc[mask]
        
        # Initialize output image
        bev = np.zeros((self.x_size, self.y_size, 4), dtype=np.float32)
        density = np.zeros((self.x_size, self.y_size), dtype=np.int32)
        
        # Aggregate point features per pixel
        for i in range(len(pc_valid)):
            x, y = x_img[i], y_img[i]
            density[x, y] += 1
            bev[x, y, 1] += pc_valid[i, 4]  # intensity
            bev[x, y, 2] = max(bev[x, y, 2], pc_valid[i, 2])  # max height
            bev[x, y, 3] += pc_valid[i, 3]  # doppler
        
        # Normalize aggregated values
        mask = density > 0
        bev[mask, 0] = np.log1p(density[mask])  # log density
        bev[mask, 1] /= density[mask]  # mean intensity
        bev[mask, 3] /= density[mask]  # mean doppler
        
        return bev


class RangeImageConverter:
    """Convert 4D radar point cloud to Range Image (spherical projection)"""
    
    def __init__(self, azimuth_bins=512, elevation_bins=64, max_range=100.0):
        """
        Args:
            azimuth_bins: number of horizontal bins (0-360 degrees)
            elevation_bins: number of vertical bins
            max_range: maximum range in meters
        """
        self.azimuth_bins = azimuth_bins
        self.elevation_bins = elevation_bins
        self.max_range = max_range
        
    def compute_spherical_coords(self, pc):
        """Compute spherical coordinates from Cartesian"""
        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
        
        # Range
        range_val = np.sqrt(x**2 + y**2 + z**2)
        
        # Azimuth: 0 to 360 degrees
        azimuth = np.arctan2(y, x)
        azimuth = (azimuth + np.pi) / (2 * np.pi)  # normalize to 0-1
        
        # Elevation: -90 to 90 degrees
        elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
        elevation = (elevation + np.pi/2) / np.pi  # normalize to 0-1
        
        return range_val, azimuth, elevation
    
    def __call__(self, pc):
        """
        Convert point cloud to Range Image
        
        Args:
            pc: (N, 5) array [x, y, z, doppler, intensity]
            
        Returns:
            range_image: (H, W, 6) array with channels:
                0: range (depth)
                1: intensity
                2: doppler velocity
                3: x coordinate
                4: y coordinate  
                5: z coordinate
        """
        if len(pc) == 0:
            return np.zeros((self.elevation_bins, self.azimuth_bins, 6), dtype=np.float32)
        
        # Compute spherical coordinates
        range_val, azimuth, elevation = self.compute_spherical_coords(pc)
        
        # Filter by range
        mask = (range_val > 0.25) & (range_val < self.max_range)
        pc = pc[mask]
        range_val = range_val[mask]
        azimuth = azimuth[mask]
        elevation = elevation[mask]
        
        if len(pc) == 0:
            return np.zeros((self.elevation_bins, self.azimuth_bins, 6), dtype=np.float32)
        
        # Convert to pixel indices
        azimuth_idx = (azimuth * self.azimuth_bins).astype(np.int32)
        elevation_idx = (elevation * self.elevation_bins).astype(np.int32)
        
        # Clip to valid range
        azimuth_idx = np.clip(azimuth_idx, 0, self.azimuth_bins - 1)
        elevation_idx = np.clip(elevation_idx, 0, self.elevation_bins - 1)
        
        # Initialize range image
        range_image = np.zeros((self.elevation_bins, self.azimuth_bins, 6), dtype=np.float32)
        
        # Fill range image (keep closest point per pixel)
        for i in range(len(pc)):
            e_idx, a_idx = elevation_idx[i], azimuth_idx[i]
            
            # If pixel is empty or new point is closer, update
            if range_image[e_idx, a_idx, 0] == 0 or range_val[i] < range_image[e_idx, a_idx, 0]:
                range_image[e_idx, a_idx, 0] = range_val[i]
                range_image[e_idx, a_idx, 1] = pc[i, 4]  # intensity
                range_image[e_idx, a_idx, 2] = pc[i, 3]  # doppler
                range_image[e_idx, a_idx, 3:6] = pc[i, :3]  # x, y, z
        
        return range_image
