import * as pdfjsLib from "pdfjs-dist";
import pdfWorker from "pdfjs-dist/build/pdf.worker.mjs?url";

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorker;

export async function processPdfFile(file, chunkSize = 800, overlap = 150) {
  const arrayBuffer = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
  const extractedChunks = [];

  // Guarantee step size is strictly positive to prevent infinite loops
  const step = Math.max(1, chunkSize - overlap);

  for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
    const page = await pdf.getPage(pageNum);
    const textContent = await page.getTextContent();
    const pageText = textContent.items.map((item) => item.str).join(" ");

    if (!pageText.trim()) continue;

    let start = 0;
    while (start < pageText.length) {
      const end = start + chunkSize;
      const chunkText = pageText.slice(start, end);

      extractedChunks.push({
        pageNumber: pageNum,
        content: chunkText,
      });

      start += step;
    }
  }

  return extractedChunks;
}