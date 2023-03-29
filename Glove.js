import fs from 'fs'
import {Matrix} from 'ml-matrix'
import numeric from 'numeric'
import Timer from 'timer-max'

class Glove {
    constructor(dataset_path){
         // Read the file contents
        const fileContents = fs.readFileSync(dataset_path, 'utf-8');

        // Split the contents into lines
        const lines = fileContents.trim().split('\n');

        // Create an object to hold the word vectors
        this.wordVectors = {};

        // Parse each line and add it to the object
        for (const line of lines) {
        const [word, ...vectorValues] = line.trim().split(' ');
        this.wordVectors[word] = vectorValues.map(parseFloat); 
        }   

        // Precompute the norms of each vector
        this.vectorNorms = {}; 
        for (const word of Object.keys(this.wordVectors)) {
        const vector = this.wordVectors[word];
        const norm = new Matrix([vector]).norm();
        this.vectorNorms[word] = norm;
}

    }

    cosineSimilarity(word1, word2,config = {full_jarowink : false}) {
        // Get the vectors and norms for each word from the this.wordVectors and this.vectorNorms objects
        const vector1 = this.wordVectors[word1];
        const vector2 = this.wordVectors[word2];
        const norm1 = this.vectorNorms[word1];
        const norm2 = this.vectorNorms[word2];
      
        if ((!vector1 || !vector2) || config.full_jarowink) {
          return Glove.jaroWinklerDistance(word1,word2)
        }
      
        // Calculate the dot product
        const dotProduct = numeric.dot(vector1, vector2);
      
        // Calculate the cosine similarity using the precomputed norms
        const cosineSimilarity = dotProduct / (norm1 * norm2);
      
        return cosineSimilarity;
      }
      
      textCosineSimilarity(text1, text2,config = {full_jarowink : false}) {
        // Tokenize the texts
        const sections1 = Glove.MasterToken(text1);
        const sections2 = Glove.MasterToken(text2);
      
        let totalSimilarity = 0;
        let similarityCount = 0;
      
        // Calculate the cosine similarity between each pair of sections
        for (let i = 0; i < sections1.length; i++) {
          const section1 = sections1[i];
          for (let j = 0; j < sections2.length; j++) {
            const section2 = sections2[j];
      
            // Calculate the cosine similarity between each pair of words
            let sectionSimilarity = 0;
            let wordCount = 0;
            for (let k = 0; k < section1.tokens.length; k++) {
              const token1 = section1.tokens[k];
              for (let l = 0; l < section2.tokens.length; l++) {
                const token2 = section2.tokens[l];
                if(token1 && token2){
                    sectionSimilarity += this.cosineSimilarity(token1, token2,config);
                    wordCount++;
                }
               
              }
            }
      
            // If there were any words in the sections, calculate the average similarity and add it to the total
            if (wordCount > 0) {
              totalSimilarity += sectionSimilarity / wordCount;
              similarityCount++;
            }
          }
        }
      
        // Calculate the average similarity between the sections
        const avgSimilarity = totalSimilarity / similarityCount;
      
        return avgSimilarity;
      }


      static jaroWinklerDistance(s1, s2) {
        // Length of the strings
        const len1 = s1.length;
        const len2 = s2.length;
      
        // The maximum distance beyond which the strings are considered not similar
        const maxDistance = Math.floor(Math.max(len1, len2) / 2) - 1;
      
        // Count of matching characters
        let matchCount = 0;
      
        // Count of transpositions
        let transpositionCount = 0;
      
        // Flags to mark if a character in a string has been matched
        let s1Matches = new Array(len1).fill(false);
        let s2Matches = new Array(len2).fill(false);
      
        // Iterate over each character in the first string
        for (let i = 0; i < len1; i++) {
          // Check for matches in the second string
          for (let j = Math.max(0, i - maxDistance); j < Math.min(len2, i + maxDistance + 1); j++) {
            // If the characters match and have not been matched before, count them as a match
            if (s1[i] === s2[j] && !s2Matches[j]) {
              s1Matches[i] = true;
              s2Matches[j] = true;
              matchCount++;
              break;
            }
          }
        }
      
        // If there are no matching characters, the distance is 0
        if (matchCount === 0) {
          return 0;
        }
      
        // Count transpositions
        let k = 0;
        for (let i = 0; i < len1; i++) {
          if (s1Matches[i]) {
            while (!s2Matches[k]) {
              k++;
            }
            if (s1[i] !== s2[k]) {
              transpositionCount++;
            }
            k++;
          }
        }
      
        // Calculate the similarity score
        const similarity = (matchCount / len1 + matchCount / len2 + (matchCount - transpositionCount / 2) / matchCount) / 3;
      
        // Apply the Jaro-Winkler boost if the strings share a common prefix
        const prefixLength = Math.min(4, Math.min(s1.length, s2.length));
        let commonPrefix = 0;
        for (let i = 0; i < prefixLength; i++) {
          if (s1[i] === s2[i]) {
            commonPrefix++;
          } else {
            break;
          }
        }
        const jaroWinklerScore = similarity + commonPrefix * 0.1 * (1 - similarity);
      
        return jaroWinklerScore;
      }
      

static MasterToken(text, config = {}) {
  const delimiterCharacters = [
    { character: ',', name: 'comma' },
    { character: '.', name: 'period' },
    { character: '?', name: 'question mark' },
    { character: '!', name: 'exclamation mark' },
    ...(config.newcharacters || []) // Merge in any additional delimiter characters from the config
  ];

  const sections = [];
  let currentSection = { tokens: [] };

  for (let i = 0; i < text.length; i++) {
    const character = text[i];
    const delimiter = delimiterCharacters.find((d) => d.character === character);

    if (delimiter) {
      // If we find a delimiter character, push the current section and start a new one
      currentSection.break_character = delimiter;
      sections.push(currentSection);
      currentSection = { tokens: [] };
    } else if (/\s/.test(character)) {
      // If we find a space character, move to the next token
      currentSection.tokens.push('');
    } else {
      // If not, add the character to the current token
      const currentToken = currentSection.tokens[currentSection.tokens.length - 1];
      currentSection.tokens[currentSection.tokens.length - 1] = currentToken + character;
    }
  }

  // Push the final section if there are any remaining tokens
  if (currentSection.tokens.length > 0) {
    sections.push(currentSection);
  }

  // Remove one-character tokens and empty tokens
  sections.forEach(section => {
    section.tokens = section.tokens.filter(token => token.length > 1);
  });

  // Remove empty sections
  return sections.filter(section => section.tokens.length > 0);
}

}

export default Glove