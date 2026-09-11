import sys
from rlls16.preview import preview

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "worlds/canonical_world"
    preview(target)
