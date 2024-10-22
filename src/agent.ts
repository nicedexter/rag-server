'use server'

import {
  HuggingFaceEmbedding,
  Ollama,
  QdrantVectorStore,
  QueryEngineTool,
  ReActAgent,
  serviceContextFromDefaults,
  Settings,
  SimpleDirectoryReader,
  VectorStoreIndex
} from "llamaindex"
import * as fs from 'fs/promises'
import { performance } from 'perf_hooks'

const DIRECTORY_PATH = "./docs"
const PARSING_CACHE = "./cache.json"

// Singleton class to initialize the agent
// transpose to function

class Agent {
  private static _instance: Agent
  private static _lock = new Promise<void>((resolve) => resolve());
  private static agent: ReActAgent

  private constructor() { }

  public static async getInstance(): Promise<ReActAgent> {
    const start = performance.now()
    await this._lock

    if (!Agent._instance) {
      Agent._instance = new Agent()
      this.agent = await this._instance.init()
    }

    const end = performance.now()
    console.log(`getInstance duration: ${end - start}ms`)

    return await this.agent
  }

  private async init() {
    const start = performance.now()

    Settings.llm = new Ollama({
      model: "llama3.2:1b",
      config: {
        host: "http://localhost:11434",
      }
    })

    Settings.embedModel = new HuggingFaceEmbedding({
      modelType: "BAAI/bge-small-en-v1.5",
      quantized: false,
    })

    Settings.callbackManager.on("llm-tool-call", (event) => {
      console.log(event.detail)
    })

    Settings.callbackManager.on("llm-tool-result", (event) => {
      console.log(event.detail)
    })

    Settings.callbackManager.on("llm-start", (event) => {
      console.log(event.detail)
    })

    Settings.callbackManager.on("agent-start", (event) => {
      console.log(event.detail)
    })

    const vectorStore = new QdrantVectorStore({
      url: "http://localhost:6333",
    })

    let cache: {[key: string]: string} = {}
    let cacheExists = false
    let files = []
    try {
      await fs.access(PARSING_CACHE, fs.constants.F_OK)
      cacheExists = true
    } catch (e) {
      console.log("No cache found")
    }

    if (cacheExists) {
      cache = JSON.parse(await fs.readFile(PARSING_CACHE, "utf-8"))
      files = await fs.readdir(DIRECTORY_PATH)
    }

    const reader = new SimpleDirectoryReader()
    const documents = await reader.loadData(DIRECTORY_PATH)
    
    await fs.writeFile(PARSING_CACHE, JSON.stringify(documents))

    const service = serviceContextFromDefaults({
      llm: Settings.llm,
      embedModel: Settings.embedModel,
    })

    const index = await VectorStoreIndex.fromDocuments(documents,
      {
        serviceContext: service,
        vectorStores: { TEXT: vectorStore }
      })

    const retriever = await index.asRetriever({ similarityTopK: 10 })
    const queryEngine = await index.asQueryEngine({ retriever })

    const tools = [
      new QueryEngineTool({
        queryEngine,
        metadata: {
          name: "chorus_research_assistant_tool",
          description: `You are a research project assistant at CHUV hospital, ensuring that the projects meet the requirements of the CHUV and you can answer detailed questions about research in CHUV hospital. You help project managers and researchers to find information about research projects.`,
        },
      }),
    ]

    const end = performance.now()
    console.log(`init duration: ${end - start}ms`)

    return new ReActAgent({ tools })
  }
}

export default Agent
