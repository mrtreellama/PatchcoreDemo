export interface DeviceInfo {
  userAgent: string;
  platform: string;
  logicalCores: number;
  deviceMemoryGb?: number;
  crossOriginIsolated: boolean;
  webgpu: boolean;
  webgpuAdapter?: Record<string, string>;
  screen: string;
}

export async function deviceInfo(): Promise<DeviceInfo> {
  const nav = navigator as Navigator & { deviceMemory?: number; gpu?: GPU; userAgentData?: { platform?: string } };
  let webgpuAdapter: Record<string, string> | undefined;
  if (nav.gpu) {
    try {
      const a = await nav.gpu.requestAdapter();
      if (a?.info) {
        webgpuAdapter = {
          vendor: a.info.vendor, architecture: a.info.architecture,
          device: a.info.device, description: a.info.description,
        };
      }
    } catch {
      /* not available */
    }
  }
  return {
    userAgent: navigator.userAgent,
    platform: nav.userAgentData?.platform ?? navigator.platform,
    logicalCores: navigator.hardwareConcurrency ?? 0,
    deviceMemoryGb: nav.deviceMemory,
    crossOriginIsolated: self.crossOriginIsolated,
    webgpu: !!nav.gpu && !!webgpuAdapter,
    webgpuAdapter,
    screen: `${screen.width}x${screen.height}@${devicePixelRatio}`,
  };
}

/** Short human label for footers, e.g. "Windows · 8 cores · NVIDIA". */
export function shortDevice(d?: DeviceInfo): string {
  if (!d) return '';
  const gpu = d.webgpuAdapter?.vendor || d.webgpuAdapter?.description;
  return [d.platform, `${d.logicalCores} cores`, gpu].filter(Boolean).join(' · ');
}
