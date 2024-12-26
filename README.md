# Path to Love

## Overview
"Path to Love" is a project that visualizes the relationship between two individuals through the words shared in their messages. By analyzing iMessage data, the project generates a co-occurrence network of words, allowing us to explore the journey of communication. The aim is to represent the evolution of love and connection through data visualization.

## Project Description
This project uses Python and several libraries to:
1. Extract message data from iMessages.
2. Clean and process the data to extract unique words.
3. Build a co-occurrence matrix to capture relationships between words.
4. Visualize the data through various graph layouts, highlighting the shortest paths between words, particularly focusing on the word "love."

The final output is a visually appealing graph representing the connections between words, symbolizing the emotional journey and shared communication between two people.

![Image](figures/path-to-love.png)

## Installation
To run the project locally, you'll need Python 3 and the following libraries:
- `imessage-tools`
- `pyEnchant`
- `networkx`
- `matplotlib`
- pandas
- numpy

### Steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/FabMougou/path-to-love.git
