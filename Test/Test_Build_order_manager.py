import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src.build_order_manager import BuildOrderManager
from src.resource_manager import ResourceManager

# Test class to test BuildOrderManager
class TestBuildOrderManager(unittest.TestCase):

    def setUp(self):
        # Initialize the ResourceManager with default settings
        self.resource_manager = ResourceManager(initial_metal=1000, initial_energy=1000, build_power=400)
        self.build_order_manager = BuildOrderManager(self.resource_manager)

        # Define phase one goals
        self.build_order_manager.phase_one_goal = False
        self.build_order_manager.phase_one_metal_goal = 20  # Example goal for metal rate
        self.build_order_manager.phase_one_energy_goal = 200  # Example goal for energy rate

    def test_create_build_order(self):
        build_order = self.build_order_manager.create_build_order()  # Create the build order

        # Check if the build order is not empty
        self.assertTrue(len(build_order) > 0, "Build order should not be empty.")

        # Check if the phase one goal is met
        self.assertTrue(self.build_order_manager.phase_one_goal, "Phase one goal should be met.")

        # Print the build order
        print("Build Order:")
        for task_info in build_order:
            task_name = task_info["task"]
            build_time = task_info["build_time"]
            current_metal = task_info.get("current_metal", "N/A")
            current_energy = task_info.get("current_energy", "N/A")
            print(f"- {task_name}: {build_time:.2f}s, Metal: {current_metal}, Energy: {current_energy}")

        # Additional assertions for various conditions
        # Check if a factory was built
        self.assertTrue(self.build_order_manager.built_factory, "Factory should be built.")
        # Check if a builder was built
        self.assertTrue(self.build_order_manager.built_builder, "Builder should be built.")
        # Check if an advanced solar was built
        self.assertTrue(self.build_order_manager.built_adv_solar, "Advanced Solar should be built.")
        # Check if metal extractor cap was reached
        self.assertTrue(
            self.build_order_manager.metal_extractor_count >= self.build_order_manager.max_metal_extractors,
            "Metal extractor cap should be reached."
        )

        # Check if energy goal was reached
        self.assertTrue(
            self.resource_manager.get_current_energy_rate() >= self.build_order_manager.phase_one_energy_goal,
            "Energy goal should be met."
        )

if __name__ == "__main__":
    unittest.main()
