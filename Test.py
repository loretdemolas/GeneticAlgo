import unittest
from BarBuildOrder import ResourceManager, BuildOrderManager, tasks, task_prerequisites


class TestResourceManager(unittest.TestCase):
    def setUp(self):
        self.resource_manager = ResourceManager(initial_metal=1000, initial_energy=1000, build_power=400)

    def test_accumulate_resources(self):
        self.resource_manager.accumulate_resources(10)
        self.assertEqual(self.resource_manager.get_current_metal(), 1000)  # No metal rate yet
        self.assertEqual(self.resource_manager.get_current_energy(), 1200)  # 20 energy rate * 10 seconds

    def test_deduct_resources(self):
        self.resource_manager.deduct_resources(100, 200)
        self.assertEqual(self.resource_manager.get_current_metal(), 900)
        self.assertEqual(self.resource_manager.get_current_energy(), 800)

    def test_update_rates(self):
        self.resource_manager.update_rates("Build Metal Extractor")
        self.assertAlmostEqual(self.resource_manager.get_current_metal_rate(), 1.8, places=1)

        self.resource_manager.update_rates("Build Solar Collector")
        self.assertEqual(self.resource_manager.get_current_energy_rate(), 40)  # 20 + 20


class TestBuildOrderManager(unittest.TestCase):
    def setUp(self):
        self.resource_manager = ResourceManager()
        self.build_manager = BuildOrderManager(self.resource_manager, max_tasks=100)

    def test_can_build(self):
        # Assuming "Build Solar Collector" has metal and energy costs <= 1000
        self.assertTrue(self.build_manager.can_build("Build Solar Collector"))

    def test_build_task(self):
        build_time = self.build_manager.build_task("Build Metal Extractor")
        self.assertGreater(build_time, 0)  # Ensure it built something
        self.assertIn("Build Metal Extractor", self.build_manager.completed_tasks)

    def test_create_build_order(self):
        build_order = self.build_manager.create_build_order()
        self.assertEqual(len(build_order), self.build_manager.max_tasks)
        print(build_order)
        # Optionally check the tasks created and order

# This command runs the test cases
if __name__ == "__main__":
    unittest.main()
