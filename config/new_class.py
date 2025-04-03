# new_activity_class.py

# Define new activity classes using the Action-Slot schema
# Each activity is described using slot labels: Subject, Action, Object, Location

new_activities = [
    {
        "id": "activity_001",
        "slots": {
            "Subject": "Person",
            "Action": "Handstand",
            "Object": None,
            "Location": "Sidewalk"
        }
    },
    {
        "id": "activity_002",
        "slots": {
            "Subject": "Dog",
            "Action": "Chasing",
            "Object": "Ball",
            "Location": "Park"
        }
    },
    {
        "id": "activity_003",
        "slots": {
            "Subject": "Person",
            "Action": "Crawling",
            "Object": None,
            "Location": "Crosswalk"
        }
    },
    {
        "id": "activity_004",
        "slots": {
            "Subject": "Cyclist",
            "Action": "Falling",
            "Object": None,
            "Location": "Intersection"
        }
    },
    {
        "id": "activity_005",
        "slots": {
            "Subject": "Animal",
            "Action": "Crossing",
            "Object": None,
            "Location": "Highway"
        }
    },
    {
        "id": "activity_006",
        "slots": {
            "Subject": "Child",
            "Action": "Running",
            "Object": "Kite",
            "Location": "Playground"
        }
    },
    {
        "id": "activity_007",
        "slots": {
            "Subject": "Skater",
            "Action": "Jumping",
            "Object": "Ramp",
            "Location": "Skatepark"
        }
    }
]


def get_new_activities():
    return new_activities


if __name__ == "__main__":
    for activity in get_new_activities():
        print(activity)
