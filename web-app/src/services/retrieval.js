import { getAllChunks } from "./db";

export async function retrieveRelevantChunks(query, k = 3) {
  const allChunks = await getAllChunks();
  if (!allChunks || allChunks.length === 0) return [];

  // Extract alphanumeric terms, ignoring empty spaces
  const queryTerms = (query.toLowerCase().match(/\w+/g) || []).filter(
    (term) => term.length > 1
  );

  if (queryTerms.length === 0) return allChunks.slice(0, k);

  const scoredChunks = allChunks.map((chunk) => {
    const text = chunk.content.toLowerCase();
    let score = 0;

    queryTerms.forEach((term) => {
      if (!term) return;
      const occurrences = text.split(term).length - 1;
      score += occurrences;
    });

    return { ...chunk, score };
  });

  const matched = scoredChunks.filter((c) => c.score > 0);

  // Fallback to top initial chunks if no term matches score > 0
  if (matched.length === 0) {
    return allChunks.slice(0, k);
  }

  return matched.sort((a, b) => b.score - a.score).slice(0, k);
}

export function formatContext(retrievedChunks) {
  if (!retrievedChunks || retrievedChunks.length === 0) {
    return "No specific context found in uploaded notes.";
  }

  return retrievedChunks
    .map((chunk) => `[Page ${chunk.pageNumber}]:\n${chunk.content}`)
    .join("\n\n");
}