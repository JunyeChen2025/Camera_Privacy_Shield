import requests, time, csv, re, sys

URL = "http://192.168.0.123:5000/video_feed"
OUT = "e2e_latency.csv"
TIMEOUT = 10

def main():
    # 关闭代理，避免把内网请求发到代理
    proxies = {"http": None, "https": None}

    print(f"Connecting to {URL} ...")
    r = requests.get(URL, stream=True, timeout=TIMEOUT, proxies=proxies)
    r.raise_for_status()

    ctype = r.headers.get("Content-Type", "")
    m = re.search(r'boundary=([-\w]+)', ctype, re.I)
    if not m:
        print("ERROR: not a multipart MJPEG stream. Content-Type:", ctype)
        sys.exit(1)
    boundary = ("--" + m.group(1)).encode()

    print("Boundary:", boundary.decode())

    buf = b""
    n_frames = 0

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["recv_ts_ms", "latency_ms"])

        for chunk in r.iter_content(chunk_size=4096):
            if not chunk:
                continue
            buf += chunk

            # 找 boundary
            while True:
                bpos = buf.find(boundary)
                if bpos < 0:
                    break
                # 丢弃 boundary 之前的内容
                buf = buf[bpos + len(boundary):]

                # 期望后面是 \r\n
                if buf[:2] == b"\r\n":
                    buf = buf[2:]

                # 找头部结束 \r\n\r\n
                h_end = buf.find(b"\r\n\r\n")
                if h_end < 0:
                    # 头还没收全
                    break

                header_blob = buf[:h_end].decode(errors="ignore")
                buf = buf[h_end + 4:]

                # 解析头；至少有 Content-Type 和 Content-Length
                headers = {}
                for line in header_blob.split("\r\n"):
                    if ":" in line:
                        k, v = line.split(":", 1)
                        headers[k.strip().lower()] = v.strip()

                # 解析我们自定义的 X-TS（纳秒）
                send_ts_ns = None
                if "x-ts" in headers:
                    try:
                        send_ts_ns = int(headers["x-ts"])
                    except Exception:
                        send_ts_ns = None

                # 解析 Content-Length
                try:
                    clen = int(headers.get("content-length", "0"))
                except Exception:
                    clen = 0

                # 读取图像体
                if len(buf) < clen:
                    # body 还不全，继续收
                    break

                jpeg = buf[:clen]
                buf = buf[clen:]

                # 一个 frame 完成
                recv_ts_ns = time.time_ns()
                if send_ts_ns is not None:
                    latency_ms = (recv_ts_ns - send_ts_ns) / 1e6
                    w.writerow([int(recv_ts_ns / 1e6), f"{latency_ms:.2f}"])
                n_frames += 1

                # 每收 30 帧提示一下
                if n_frames % 30 == 0:
                    print(f"frames: {n_frames} ...")

if __name__ == "__main__":
    main()
