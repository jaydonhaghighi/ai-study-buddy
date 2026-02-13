type AcquireResult = {
  stream: MediaStream;
  release: () => void;
};

let activeStream: MediaStream | null = null;
let refCount = 0;
let pending: Promise<MediaStream> | null = null;

async function getOrCreateStream(): Promise<MediaStream> {
  if (activeStream) return activeStream;
  if (pending) return pending;

  pending = navigator.mediaDevices
    .getUserMedia({ video: true, audio: false })
    .then((stream) => {
      activeStream = stream;
      return stream;
    })
    .finally(() => {
      pending = null;
    });

  return pending;
}

function stopStream(stream: MediaStream) {
  stream.getTracks().forEach((t) => t.stop());
}

export async function acquireWebcamStream(): Promise<AcquireResult> {
  const stream = await getOrCreateStream();
  refCount += 1;

  let released = false;
  const release = () => {
    if (released) return;
    released = true;
    refCount = Math.max(0, refCount - 1);
    if (refCount === 0 && activeStream) {
      const s = activeStream;
      activeStream = null;
      stopStream(s);
    }
  };

  return { stream, release };
}

