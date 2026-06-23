"""
Wanted-person watchlist.

Put face images in the watchlist folder (default: data/watchlist/), one person
per image, filename = person's name (e.g. "ramesh.jpg"). When a matching face
appears in the feed, an alert is raised.

Note: face matching works well when faces are reasonably large in the frame
(entry / gate / moderate scenes). It does NOT scale to thousands of tiny faces
in a dense crowd -- that is a different problem.
"""

import os

import cv2

try:
    import face_recognition
    _HAVE_FR = True
except Exception as e:  # pragma: no cover
    _HAVE_FR = False
    print(f"[WARN] face_recognition unavailable, watchlist disabled: {e}")


class Watchlist:
    def __init__(self, watch_dir, tolerance=0.5, downscale=0.5, cooldown=5.0):
        self.watch_dir = watch_dir
        self.tolerance = tolerance
        self.downscale = downscale          # process a smaller frame for speed
        self.cooldown = cooldown            # seconds between repeat alerts per name
        self.encodings = []
        self.names = []
        self._last_alert = {}               # name -> last alert time
        self.enabled = _HAVE_FR
        if self.enabled:
            self._load()

    def _load(self):
        if not os.path.isdir(self.watch_dir):
            os.makedirs(self.watch_dir, exist_ok=True)
            print(f"[INFO] Watchlist folder created (empty): {self.watch_dir}")
            return
        for fn in os.listdir(self.watch_dir):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(self.watch_dir, fn)
            try:
                img = face_recognition.load_image_file(path)
                encs = face_recognition.face_encodings(img)
                if encs:
                    self.encodings.append(encs[0])
                    self.names.append(os.path.splitext(fn)[0])
                    print(f"[INFO] Watchlist loaded: {fn}")
                else:
                    print(f"[WARN] No face found in watchlist image: {fn}")
            except Exception as e:
                print(f"[WARN] Failed to load watchlist image {fn}: {e}")
        print(f"[INFO] Watchlist active with {len(self.names)} person(s).")

    def check(self, frame, timestamp):
        """Return list of matches: [{name, box=(x1,y1,x2,y2)}]. Respects cooldown."""
        if not self.enabled or not self.encodings:
            return []
        small = cv2.resize(frame, (0, 0), fx=self.downscale, fy=self.downscale)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, locations)
        inv = 1.0 / self.downscale

        matches = []
        for (top, right, bottom, left), enc in zip(locations, encodings):
            dists = face_recognition.face_distance(self.encodings, enc)
            if len(dists) == 0:
                continue
            best = int(dists.argmin())
            if dists[best] <= self.tolerance:
                name = self.names[best]
                last = self._last_alert.get(name, 0)
                if timestamp - last < self.cooldown:
                    continue
                self._last_alert[name] = timestamp
                box = (int(left * inv), int(top * inv),
                       int(right * inv), int(bottom * inv))
                matches.append({"name": name, "box": box})
        return matches
