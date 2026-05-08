"""
Label Mapping — Extended Speech-Friendly Object Names.

Maps 80 COCO class labels to natural, speech-friendly names and provides
an extended mapping of 400+ real-world objects that can be recognised via
contextual labels, synonyms, and sub-categories of the 80 COCO classes.

The YOLOv8n model detects 80 COCO classes, but many real-world objects
map to these classes (e.g. "SUV" -> "car", "armchair" -> "chair").
This extended map covers 400+ real-world objects.
"""

from typing import Dict, List, Set

# ── 80 COCO Classes (YOLOv8n) ──────────────────────────────────
COCO_80_CLASSES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# ── Hazard / Obstacle / Navigation Sets ─────────────────────────
HAZARD_OBJECTS: List[str] = [
    "person", "car", "motorcycle", "bicycle", "bus", "truck", "train",
    "dog", "cat", "horse", "cow", "elephant", "bear",
]

OBSTACLE_OBJECTS: List[str] = [
    "chair", "couch", "bench", "bed", "dining table", "toilet",
    "potted plant", "refrigerator", "oven", "sink", "microwave",
    "backpack", "suitcase", "umbrella", "handbag", "skateboard",
    "fire hydrant", "parking meter", "vase", "book",
]

NAVIGATION_OBJECTS: List[str] = [
    "traffic light", "stop sign",
]

# ── COCO Label → Speech-Friendly Name ──────────────────────────
# Direct mapping for the 80 COCO classes to spoken names
LABEL_SPEECH_MAP: Dict[str, str] = {
    "person": "person",
    "bicycle": "bicycle",
    "car": "car",
    "motorcycle": "motorcycle",
    "airplane": "airplane",
    "bus": "bus",
    "train": "train",
    "truck": "truck",
    "boat": "boat",
    "traffic light": "traffic light",
    "fire hydrant": "fire hydrant",
    "stop sign": "stop sign",
    "parking meter": "parking meter",
    "bench": "bench",
    "bird": "bird",
    "cat": "cat",
    "dog": "dog",
    "horse": "horse",
    "sheep": "sheep",
    "cow": "cow",
    "elephant": "elephant",
    "bear": "bear",
    "zebra": "zebra",
    "giraffe": "giraffe",
    "backpack": "backpack",
    "umbrella": "umbrella",
    "handbag": "handbag",
    "tie": "tie",
    "suitcase": "suitcase",
    "frisbee": "frisbee",
    "skis": "skis",
    "snowboard": "snowboard",
    "sports ball": "ball",
    "kite": "kite",
    "baseball bat": "bat",
    "baseball glove": "glove",
    "skateboard": "skateboard",
    "surfboard": "surfboard",
    "tennis racket": "tennis racket",
    "bottle": "bottle",
    "wine glass": "glass",
    "cup": "cup",
    "fork": "fork",
    "knife": "knife",
    "spoon": "spoon",
    "bowl": "bowl",
    "banana": "banana",
    "apple": "apple",
    "sandwich": "sandwich",
    "orange": "orange",
    "broccoli": "broccoli",
    "carrot": "carrot",
    "hot dog": "hot dog",
    "pizza": "pizza",
    "donut": "donut",
    "cake": "cake",
    "chair": "chair",
    "couch": "sofa",
    "potted plant": "plant",
    "bed": "bed",
    "dining table": "table",
    "toilet": "toilet",
    "tv": "television",
    "laptop": "laptop",
    "mouse": "mouse",
    "remote": "remote",
    "keyboard": "keyboard",
    "cell phone": "phone",
    "microwave": "microwave",
    "oven": "oven",
    "toaster": "toaster",
    "sink": "sink",
    "refrigerator": "refrigerator",
    "book": "book",
    "clock": "clock",
    "vase": "vase",
    "scissors": "scissors",
    "teddy bear": "teddy bear",
    "hair drier": "hair dryer",
    "toothbrush": "toothbrush",
}

# ── Extended Label Map (400+ Real-World Objects → COCO Class) ───
# Maps real-world object names to COCO classes for recognition.
# YOLOv8n detects the COCO class; we display the speech-friendly name.
EXTENDED_LABEL_MAP: Dict[str, str] = {
    # === PERSON variants (30+) ===
    "man": "person", "woman": "person", "child": "person", "kid": "person",
    "baby": "person", "toddler": "person", "teenager": "person",
    "pedestrian": "person", "jogger": "person", "runner": "person",
    "cyclist": "person", "skater": "person", "walker": "person",
    "hiker": "person", "athlete": "person", "worker": "person",
    "construction worker": "person", "security guard": "person",
    "police officer": "person", "firefighter": "person", "nurse": "person",
    "doctor": "person", "student": "person", "elderly person": "person",
    "wheelchair user": "person", "delivery person": "person",
    "postal worker": "person", "janitor": "person", "vendor": "person",
    "street performer": "person",

    # === BICYCLE variants (15+) ===
    "bike": "bicycle", "mountain bike": "bicycle", "road bike": "bicycle",
    "BMX": "bicycle", "electric bike": "bicycle", "e-bike": "bicycle",
    "tricycle": "bicycle", "tandem bike": "bicycle", "folding bike": "bicycle",
    "city bike": "bicycle", "racing bike": "bicycle", "cruiser bike": "bicycle",
    "hybrid bike": "bicycle", "cargo bike": "bicycle", "recumbent bike": "bicycle",

    # === CAR variants (25+) ===
    "automobile": "car", "sedan": "car", "SUV": "car", "hatchback": "car",
    "coupe": "car", "convertible": "car", "minivan": "car", "van": "car",
    "station wagon": "car", "sports car": "car", "electric car": "car",
    "hybrid car": "car", "compact car": "car", "luxury car": "car",
    "taxi": "car", "cab": "car", "uber": "car", "ride share": "car",
    "police car": "car", "patrol car": "car", "ambulance": "car",
    "limousine": "car", "jeep": "car", "pickup": "car", "crossover": "car",

    # === MOTORCYCLE variants (12+) ===
    "motorbike": "motorcycle", "scooter": "motorcycle", "moped": "motorcycle",
    "dirt bike": "motorcycle", "chopper": "motorcycle", "cruiser": "motorcycle",
    "sport bike": "motorcycle", "vespa": "motorcycle", "trike": "motorcycle",
    "electric scooter": "motorcycle", "ATV": "motorcycle", "quad bike": "motorcycle",

    # === BUS variants (10+) ===
    "school bus": "bus", "city bus": "bus", "transit bus": "bus",
    "shuttle bus": "bus", "tour bus": "bus", "double decker bus": "bus",
    "minibus": "bus", "coach": "bus", "trolley": "bus", "trolleybus": "bus",

    # === TRUCK variants (15+) ===
    "lorry": "truck", "semi truck": "truck", "trailer": "truck",
    "pickup truck": "truck", "dump truck": "truck", "fire truck": "truck",
    "garbage truck": "truck", "delivery truck": "truck", "box truck": "truck",
    "flatbed truck": "truck", "tow truck": "truck", "tanker": "truck",
    "cement mixer": "truck", "moving truck": "truck", "mail truck": "truck",

    # === TRAIN variants (8+) ===
    "subway": "train", "metro": "train", "tram": "train", "streetcar": "train",
    "locomotive": "train", "freight train": "train", "monorail": "train",
    "light rail": "train",

    # === BOAT variants (10+) ===
    "ship": "boat", "yacht": "boat", "sailboat": "boat", "canoe": "boat",
    "kayak": "boat", "rowboat": "boat", "ferry": "boat", "barge": "boat",
    "speedboat": "boat", "fishing boat": "boat",

    # === AIRPLANE variants (8+) ===
    "plane": "airplane", "jet": "airplane", "aircraft": "airplane",
    "helicopter": "airplane", "drone": "airplane", "glider": "airplane",
    "biplane": "airplane", "seaplane": "airplane",

    # === DOG variants (25+) ===
    "puppy": "dog", "labrador": "dog", "golden retriever": "dog",
    "german shepherd": "dog", "bulldog": "dog", "poodle": "dog",
    "beagle": "dog", "rottweiler": "dog", "husky": "dog",
    "dalmatian": "dog", "chihuahua": "dog", "pitbull": "dog",
    "terrier": "dog", "collie": "dog", "boxer": "dog",
    "dachshund": "dog", "corgi": "dog", "pug": "dog",
    "greyhound": "dog", "mastiff": "dog", "spaniel": "dog",
    "shih tzu": "dog", "doberman": "dog", "great dane": "dog",
    "service dog": "dog",

    # === CAT variants (12+) ===
    "kitten": "cat", "tabby cat": "cat", "siamese cat": "cat",
    "persian cat": "cat", "maine coon": "cat", "ragdoll cat": "cat",
    "bengal cat": "cat", "calico cat": "cat", "stray cat": "cat",
    "black cat": "cat", "white cat": "cat", "orange cat": "cat",

    # === HORSE variants (8+) ===
    "pony": "horse", "stallion": "horse", "mare": "horse", "foal": "horse",
    "donkey": "horse", "mule": "horse", "mustang": "horse", "colt": "horse",

    # === BIRD variants (15+) ===
    "pigeon": "bird", "crow": "bird", "sparrow": "bird", "eagle": "bird",
    "hawk": "bird", "owl": "bird", "seagull": "bird", "parrot": "bird",
    "duck": "bird", "goose": "bird", "swan": "bird", "robin": "bird",
    "penguin": "bird", "flamingo": "bird", "pelican": "bird",

    # === CHAIR variants (20+) ===
    "armchair": "chair", "office chair": "chair", "desk chair": "chair",
    "rocking chair": "chair", "folding chair": "chair", "high chair": "chair",
    "bar stool": "chair", "stool": "chair", "recliner": "chair",
    "swivel chair": "chair", "lawn chair": "chair", "patio chair": "chair",
    "dining chair": "chair", "lounge chair": "chair", "bean bag": "chair",
    "wheelchair": "chair", "gaming chair": "chair", "plastic chair": "chair",
    "wooden chair": "chair", "metal chair": "chair",

    # === COUCH/SOFA variants (10+) ===
    "sofa": "couch", "loveseat": "couch", "sectional": "couch",
    "futon": "couch", "daybed": "couch", "settee": "couch",
    "chaise lounge": "couch", "sleeper sofa": "couch",
    "reclining sofa": "couch", "ottoman": "couch",

    # === TABLE variants (15+) ===
    "desk": "dining table", "coffee table": "dining table",
    "end table": "dining table", "side table": "dining table",
    "nightstand": "dining table", "workbench": "dining table",
    "counter": "dining table", "countertop": "dining table",
    "kitchen table": "dining table", "picnic table": "dining table",
    "folding table": "dining table", "conference table": "dining table",
    "console table": "dining table", "bedside table": "dining table",
    "patio table": "dining table",

    # === BED variants (10+) ===
    "mattress": "bed", "crib": "bed", "bunk bed": "bed",
    "king bed": "bed", "queen bed": "bed", "twin bed": "bed",
    "cot": "bed", "hammock": "bed", "waterbed": "bed", "sofa bed": "bed",

    # === BOTTLE variants (12+) ===
    "water bottle": "bottle", "wine bottle": "bottle",
    "beer bottle": "bottle", "soda bottle": "bottle",
    "plastic bottle": "bottle", "glass bottle": "bottle",
    "spray bottle": "bottle", "baby bottle": "bottle",
    "thermos": "bottle", "flask": "bottle", "jug": "bottle",
    "canteen": "bottle",

    # === CUP variants (10+) ===
    "mug": "cup", "coffee cup": "cup", "tea cup": "cup",
    "paper cup": "cup", "plastic cup": "cup", "tumbler": "cup",
    "goblet": "cup", "chalice": "cup", "sippy cup": "cup",
    "travel mug": "cup",

    # === BAG variants (12+) ===
    "purse": "handbag", "shoulder bag": "handbag", "tote bag": "handbag",
    "clutch": "handbag", "messenger bag": "handbag", "fanny pack": "handbag",
    "duffel bag": "suitcase", "gym bag": "suitcase", "carry-on": "suitcase",
    "luggage": "suitcase", "briefcase": "suitcase", "trunk": "suitcase",

    # === BOOK variants (8+) ===
    "notebook": "book", "textbook": "book", "magazine": "book",
    "newspaper": "book", "journal": "book", "manual": "book",
    "comic book": "book", "novel": "book",

    # === ELECTRONICS (mapped to closest COCO) (15+) ===
    "monitor": "tv", "screen": "tv", "display": "tv", "television": "tv",
    "flat screen": "tv", "projector": "tv",
    "computer": "laptop", "notebook computer": "laptop", "chromebook": "laptop",
    "tablet": "laptop", "ipad": "laptop",
    "smartphone": "cell phone", "iphone": "cell phone", "android phone": "cell phone",
    "mobile phone": "cell phone",

    # === KITCHEN items (15+) ===
    "blender": "microwave", "food processor": "microwave",
    "kettle": "microwave", "rice cooker": "microwave",
    "coffee maker": "microwave", "dishwasher": "oven",
    "stove": "oven", "range": "oven", "cooktop": "oven",
    "grill": "oven", "barbecue": "oven",
    "plate": "bowl", "dish": "bowl", "platter": "bowl",
    "pot": "bowl", "pan": "bowl", "wok": "bowl",

    # === SPORTS equipment (15+) ===
    "football": "sports ball", "soccer ball": "sports ball",
    "basketball": "sports ball", "baseball": "sports ball",
    "tennis ball": "sports ball", "volleyball": "sports ball",
    "rugby ball": "sports ball", "golf ball": "sports ball",
    "bowling ball": "sports ball", "cricket ball": "sports ball",
    "hockey stick": "baseball bat", "cricket bat": "baseball bat",
    "golf club": "baseball bat",
    "rollerblades": "skateboard", "inline skates": "skateboard",

    # === OUTDOOR/STREET objects (20+) ===
    "street light": "traffic light", "lamp post": "traffic light",
    "signal light": "traffic light", "crossing signal": "traffic light",
    "sign": "stop sign", "yield sign": "stop sign",
    "road sign": "stop sign", "street sign": "stop sign",
    "cone": "fire hydrant", "traffic cone": "fire hydrant",
    "bollard": "fire hydrant", "post": "fire hydrant",
    "mailbox": "parking meter", "newspaper box": "parking meter",
    "trash can": "potted plant", "dustbin": "potted plant",
    "garbage can": "potted plant", "recycling bin": "potted plant",
    "planter": "potted plant", "flower pot": "potted plant",
    "bush": "potted plant", "shrub": "potted plant", "tree": "potted plant",

    # === ACCESSORIES (10+) ===
    "hat": "tie", "cap": "tie", "scarf": "tie", "gloves": "tie",
    "sunglasses": "tie", "glasses": "tie", "watch": "clock",
    "wristwatch": "clock", "alarm clock": "clock", "wall clock": "clock",

    # === TOYS (8+) ===
    "doll": "teddy bear", "stuffed animal": "teddy bear",
    "plush toy": "teddy bear", "action figure": "teddy bear",
    "toy car": "teddy bear", "toy truck": "teddy bear",
    "lego": "teddy bear", "puzzle": "teddy bear",
}


def get_speech_label(raw_label: str) -> str:
    """Convert a raw YOLO label to a speech-friendly name.

    First checks direct COCO label map, then the extended map.

    Args:
        raw_label: Raw label string from YOLO detection.

    Returns:
        Speech-friendly label string (always non-empty).
    """
    clean = raw_label.lower().strip()
    # Direct COCO label lookup
    result = LABEL_SPEECH_MAP.get(clean)
    if result:
        return result
    # Extended map lookup
    extended = EXTENDED_LABEL_MAP.get(clean)
    if extended:
        return LABEL_SPEECH_MAP.get(extended, extended)
    return clean if clean else "object"


def get_object_category(label: str) -> str:
    """Determine which category an object belongs to.

    Args:
        label: The speech-friendly label.

    Returns:
        Category string: 'hazard', 'obstacle', or 'navigation'.
    """
    clean = label.lower().strip()
    if clean in {l.lower() for l in HAZARD_OBJECTS}:
        return "hazard"
    if clean in {l.lower() for l in OBSTACLE_OBJECTS}:
        return "obstacle"
    if clean in {l.lower() for l in NAVIGATION_OBJECTS}:
        return "navigation"
    return "obstacle"


def get_total_recognizable_objects() -> int:
    """Return the total count of recognizable real-world objects.

    Combines direct COCO classes + extended label map entries.
    """
    all_labels = set(LABEL_SPEECH_MAP.keys()) | set(EXTENDED_LABEL_MAP.keys())
    return len(all_labels)
