import requests, io, time, csv, sys
import numpy as np
import cv2

URL = "http://192.168.0.123:5000/video_feed"
OUT = "privacy_metrics.csv"
N   = 400           # Sampling frame rate
TIMEOUT = 10

def parse_mjpeg(stream_bytes):
    # Parse JPEG frames from multipart/x-mixed-replace
    soi = stream_bytes.find(b'\xff\xd8')
    eoi = stream_bytes.find(b'\xff\xd9')
    if soi != -1 and eoi != -1 and eoi > soi:
        jpg = stream_bytes[soi:eoi+2]
        rest = stream_bytes[eoi+2:]
        return jpg, rest
    return None, stream_bytes

def entropy(gray):
    hist = cv2.calcHist([gray],[0],None,[256],[0,256]).ravel()
    p = hist / (gray.size + 1e-9)
    p = p[p>0]
    return float(-(p*np.log2(p)).sum())

def neighbor_corr(gray, axis=1):
    # axis=1 horizontal；axis=0 vertical
    if axis==1:
        a = gray[:, :-1].astype(np.float32)
        b = gray[:, 1: ].astype(np.float32)
    else:
        a = gray[:-1, :].astype(np.float32)
        b = gray[1:  , :].astype(np.float32)
    a-=a.mean(); b-=b.mean()
    denom = (np.sqrt((a*a).sum())*np.sqrt((b*b).sum())+1e-9)
    return float((a*b).sum()/denom)

def lap_var(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def main():
    print(f"Connecting to {URL} ...")
    proxies = {'http': None, 'https': None}
    r = requests.get(URL, stream=True, timeout=TIMEOUT, proxies=proxies)
    r.raise_for_status()

    buf = b""
    cnt = 0
    last_log = time.time()
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts_ms","entropy","corr_h","corr_v","lap_var"])
        for chunk in r.iter_content(chunk_size=4096):
            if not chunk: break
            buf += chunk
            img_bytes, buf = parse_mjpeg(buf)
            if img_bytes is None:
                continue
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            e  = entropy(gray)
            ch = neighbor_corr(gray, 1)
            cv = neighbor_corr(gray, 0)
            lv = lap_var(gray)

            w.writerow([int(time.time()*1000), f"{e:.4f}", f"{ch:.4f}", f"{cv:.4f}", f"{lv:.2f}"])
            cnt += 1
            if time.time()-last_log > 1:
                print(f"frames: {cnt}  (entropy {e:.2f}, corr_h {ch:.2f}, corr_v {cv:.2f}, lap {lv:.1f})")
                last_log = time.time()
            if cnt >= N:
                break
    print(f"Done. Metrics saved to {OUT}")

if __name__ == "__main__":
    main()
