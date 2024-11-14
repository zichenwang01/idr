import torch
import torch.nn as nn
from utils import rend_util

class DiffRayTracer:
    def __init__(self, sdf_threshold, sphere_tracing_iters, line_step_iters, line_search_step):
        self.sdf_threshold = sdf_threshold
        self.sphere_tracing_iters = sphere_tracing_iters
        self.line_step_iters = line_step_iters
        self.line_search_step = line_search_step

    def sphere_tracing(self, batch_size, num_pixels, sdf, cam_loc, ray_directions, mask_intersect, sphere_intersections):

        sphere_intersections_points = cam_loc.reshape(batch_size, 1, 1, 3) + sphere_intersections.unsqueeze(-1) * ray_directions.unsqueeze(2)
        unfinished_mask = mask_intersect.reshape(-1).clone()

        # Initialize current points
        curr_points = torch.zeros(batch_size * num_pixels, 3).cuda().float()
        curr_points[unfinished_mask] = sphere_intersections_points[:, :, 0, :].reshape(-1, 3)[unfinished_mask]
        
        # Initialize accumulated distance
        acc_dis = torch.zeros(batch_size * num_pixels).cuda().float()
        acc_dis[unfinished_mask] = sphere_intersections.reshape(-1, 2)[unfinished_mask, 0]

        # Initialize next sdf
        next_sdf = torch.zeros_like(acc_dis).cuda()
        next_sdf[unfinished_mask] = sdf(curr_points[unfinished_mask])

        # Main sphere tracing loop
        iters = 0
        while True:
            # Update sdf
            curr_sdf = torch.zeros_like(acc_dis).cuda()
            curr_sdf[unfinished_mask] = next_sdf[unfinished_mask]
            curr_sdf[curr_sdf <= self.sdf_threshold] = 0

            # Update masks
            unfinished_mask = unfinished_mask & (curr_sdf > self.sdf_threshold)

            # Check exit condition
            if unfinished_mask.sum() == 0:
                break  
            if iters >= self.sphere_tracing_iters:
                break
            iters += 1

            # Make step
            # Update distance
            acc_dis = acc_dis + curr_sdf

            # Update points
            curr_points = (cam_loc.unsqueeze(1) + acc_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)

            # Fix points which wrongly crossed the surface
            next_sdf = torch.zeros_like(acc_dis).cuda()
            next_sdf[unfinished_mask] = sdf(curr_points[unfinished_mask])

            not_projected = next_sdf < 0
            not_proj_iters = 0
            while not_projected.sum() > 0 and not_proj_iters < self.line_step_iters:
                # Update distance to step back
                acc_dis[not_projected] -= ((1 - self.line_search_step) / (2 ** not_proj_iters)) * curr_sdf[not_projected]
                
                # Update points
                curr_points[not_projected] = (cam_loc.unsqueeze(1) + acc_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)[not_projected]

                # Update sdf
                next_sdf[not_projected] = sdf(curr_points[not_projected])

                # Update mask
                not_projected = next_sdf < 0
                not_proj_iters += 1

        return curr_points, unfinished_mask, acc_dis

    def minimal_sdf_points(self, num_pixels, sdf, cam_loc, ray_directions, mask, min_dis, max_dis):
        ''' Find the minimal SDF points along the rays '''

        # Initialize points and distances
        min_mask_points = torch.zeros(num_pixels, 3).cuda().float()
        min_mask_dist = torch.zeros(num_pixels).cuda().float()

        # Iterate to find minimal SDF points
        for i in range(self.sphere_tracing_iters):
            curr_points = (cam_loc.unsqueeze(1) + min_mask_dist.reshape(-1, 1) * ray_directions).reshape(-1, 3)
            curr_sdf = sdf(curr_points)
            min_mask_dist = torch.where(curr_sdf < min_mask_dist, curr_sdf, min_mask_dist)
            min_mask_points = torch.where(curr_sdf < min_mask_dist.unsqueeze(-1), curr_points, min_mask_points)

        return min_mask_points, min_mask_dist

    def forward(self, sdf, cam_loc, object_mask, ray_directions):

        batch_size, num_pixels, _ = ray_directions.shape

        # Bounding box intersection
        sphere_intersections, mask_intersect = rend_util.get_sphere_intersection(cam_loc, ray_directions, r=self.object_bounding_sphere)

        # Sphere tracing
        curr_points, unfinished_mask, acc_dis = self.sphere_tracing(
            batch_size, num_pixels, sdf, cam_loc, ray_directions, mask_intersect, sphere_intersections
        )

        # Find minimal SDF points
        min_mask_points, min_mask_dist = self.minimal_sdf_points(num_pixels, sdf, cam_loc, ray_directions, unfinished_mask, acc_dis, acc_dis)

        # Update current points and distances
        curr_points[unfinished_mask] = min_mask_points
        acc_dis[unfinished_mask] = min_mask_dist

        return curr_points, unfinished_mask, acc_dis

# Example usage
# tracer = RayTracer(sdf_threshold=0.01, sphere_tracing_iters=100, line_step_iters=10, line_search_step=0.5)
# curr_points, unfinished_mask, acc_dis = tracer.forward(
#     batch_size, num_pixels, sdf, cam_loc, ray_directions, mask_intersect, sphere_intersections
# )