declare module 'pdf-parse' {
  export class PDFParse {
    constructor(options: { data: Buffer | Uint8Array | ArrayBuffer });
    getText(options?: { parsePageInfo?: boolean }): Promise<{ text: string }>;
    destroy(): void;
  }
}

declare module 'xlsx' {
  const XLSX: any;
  export default XLSX;
}
