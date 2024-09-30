class Structure:
    def __init__(self, bio_structure):
        self.bio_structure = bio_structure
        self.id = bio_structure.id

    def __getitem__(self, key):
        return self.bio_structure[key]

    def __iter__(self):
        return iter(self.bio_structure)
