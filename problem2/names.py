from faker import Faker

# Indian locale
fake = Faker("en_IN")

names = set()

while len(names) < 1000:
    name = fake.name()   # full name (first + surname)
    names.add(name)

with open("TrainingNames.txt", "w") as f:
    for name in names:
        f.write(name + "\n")

print("TrainingNames.txt generated with", len(names), "names")