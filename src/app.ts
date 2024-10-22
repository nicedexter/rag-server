import express, { Request, Response } from 'express'
import Agent from './agent'
import { ReActAgent, EngineResponse } from 'llamaindex'
import cors from 'cors'

let researchAssistant: ReActAgent

(async () => {
  researchAssistant = await Agent.getInstance()
})()

const app = express()
app.use(express.json())
app.use(cors())

app.get('/', (req: Request, res: Response) => {
  res.send('Hello World')
})

app.post('/', async (req: Request, res: Response) => {

  if (!researchAssistant) {
    res.send('Agent not initialized')
    return
  }

  res.writeHead(200, {
    'Content-Type': 'text/plain',
    'Transfer-Encoding': 'chunked'
  })

  const { message } = req.body
  const stream = await researchAssistant.chat({
    stream: true,
    message,
  })

  for await (const chunk of stream) {
    res.write(chunk.delta || "")
  }

  res.end()
})

app.listen(3300, () => {
  console.log('Server started on port 3300')
})
