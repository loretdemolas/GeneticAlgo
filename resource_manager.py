
class ResourceManager:
    def __init__(self, initial_metal=1000, initial_energy=1000, build_power=400):
        self.current_metal = initial_metal
        self.current_energy = initial_energy
        self.build_power = build_power
        self.metal_rate = 0
        self.energy_rate = 20

    def accumulate_resources(self, time_interval):
        self.current_metal += self.metal_rate * time_interval
        self.current_energy += self.energy_rate * time_interval

    def deduct_resources(self, metal, energy):
        self.current_metal -= metal
        self.current_energy -= energy

    def get_build_power(self):
        return self.build_power

    def get_current_metal(self):
        return self.current_metal

    def get_current_energy(self):
        return self.current_energy

    def get_current_metal_rate(self):
        return self.metal_rate

    def get_current_energy_rate(self):
        return self.energy_rate

    def update_rates(self, task_name):
        if task_name == "Build Metal Extractor":
            self.metal_rate += 1.8
        elif task_name == "Build Solar Collector":
            self.energy_rate += 20
        elif task_name == "Build Energy Converter":
            self.energy_rate -= 70
            self.metal_rate += 1
        elif task_name == "Build Builder":
            self.build_power += 100
        elif task_name == "Build Advanced Solar Collector":
            self.energy_rate += 75
        elif task_name == "Build Advanced Metal Extractor":
            self.metal_rate += 8
        elif task_name == "Build Advanced Energy Converter":
            self.energy_rate -= 600
            self.metal_rate += 10.3

