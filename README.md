# Horizon Bench
Horizon Bench contains reproductions of AI Scientist projects. 

## Goals with Horizon Bench
1. Provide easy-to-run, reproducible implementations of various AI Scientist projects. Research code is rarely directly runnable, and often requires hours of debugging to get it to work.
2. Provide a common framework (same LLMs, same evaluation method) for comparing different AI Scientist projects.
3. Provide a shared code backbone, to enable quick reimplementation of future papers on AI Scientists.

## Installation
`conda env create --name horizon_bench python=3.13 -c conda-forge`  
`conda activate horizon_bench`
`pip install -r requirements.txt`   
Pandoc is used for COI Agent's paper text extraction
`sudo apt install pandoc`

### GROBID (required for OpenReview PDF text extraction)
GROBID is used via `scipdf-parser` to extract full text from OpenReview PDFs.
Requires Java 17 (GROBID's Gradle wrapper does not support Java 21+).

1. Install Java 17 and scipdf-parser:
   ```bash
   sudo apt install -y openjdk-17-jdk
   pip install scipdf-parser
   ```
2. Install GROBID (check which version is currently newest, 0.8.2 at time of writing):
   ```bash
   wget https://github.com/kermitt2/grobid/archive/0.8.2.zip
   unzip 0.8.2.zip
   cd grobid-0.8.2
   ./gradlew clean install
   ```
3. Run GROBID before using the PDF extraction:
   ```bash
   cd grobid-0.8.2
   ./gradlew run
   ```
   GROBID will start on `http://localhost:8070` by default.

Replace the workspace path in `src/horizon_bench/Config.py` according to your setup.  
I used absolute paths to avoid conflicts.  

## Running the AI Researcher Pipeline
`jupyter notebook src/ai_researcher/reproducing_ai_researcher.ipynb`  
Word of caution: AI Researcher enables arbitrary code execution on your machine by the LLM. As far as I can tell it didn't do anything harmful yet. If you want to be extra safe, run my code in a VM.

## Running the AI Scientist Pipeline
`jupyter notebook src/ai_scientist/reproducing_ai_scientist.ipynb`

## LLM as a judge
I reimplemented AI Scientist's LLM as a judge functionality, which aggregates multiple reviews from an LLM into a final review.  
`src/ai_scientist/llm_as_a_judge/llm_as_a_judge.py`  

# Limitations of the reimplementations
I list here the simplifications that I made for the different projects. For example, I often found asyncio code which was completely redundant, because people simply always called `await` on each function. At that point, there is no concurrency benefit at all (AI Researcher, COI Agent).  


## COI Agent
### Replacing SciPDF + grobid with AI Researcher's arxiv tex download + pandoc conversion
COI Agent uses SciPDF and grobid to extract text from PDFs. This is general, as every paper has a PDF.  
However, grobid produces a lot of noisy text and some errors. For now, I replaced this with: 
1. Download latex source from arxiv
2. Convert tex to txt using pandoc
This produces much cleaner text. The downside is that this only works for arxiv papers.

<!-- May 31, 2024 -->
