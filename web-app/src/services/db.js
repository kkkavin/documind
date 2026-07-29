import Dexie from "dexie";

export const db = new Dexie("StudySyncDB");

db.version(1).stores({
  documents: "++id, name, uploadedAt",
  chunks: "++id, docId, pageNumber, content",
  chats: "++id, title, createdAt",
  messages: "++id, chatId, role, content, timestamp",
});

export async function saveDocumentWithChunks(fileName, chunks) {
  return await db.transaction("rw", db.documents, db.chunks, async () => {
    // Clear old documents if single-doc mode
    await db.documents.clear();
    await db.chunks.clear();

    const docId = await db.documents.add({
      name: fileName,
      uploadedAt: new Date(),
    });

    const chunkRecords = chunks.map((chunk) => ({
      docId,
      pageNumber: chunk.pageNumber,
      content: chunk.content,
    }));

    await db.chunks.bulkAdd(chunkRecords);
    return docId;
  });
}

export async function getAllChunks() {
  return await db.chunks.toArray();
}