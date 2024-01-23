import { NextRequest } from "next/server";
import { PineconeClient, Pinecone } from "@pinecone-database/pinecone";
//import { PineconeStore } from "langchain/vectorstores/pinecone";
import { PineconeStore} from "@langchain/pinecone";
// import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { OpenAIEmbeddings, OpenAI } from "@langchain/openai";
//import { OpenAI } from "langchain/llms/openai";
import { VectorDBQAChain, RetrievalQAChain } from "langchain/chains";
import { StreamingTextResponse, LangChainStream } from "ai";
// import { CallbackManager } from "langchain/callbacks";
import { CallbackManager } from "@langchain/core/callbacks/manager"

export const POST = async(request: NextRequest)=>{
    //Parse the POST request's JSON body
    const body = await request.json();

    //Use Vercel's `ai` package to setup a stream
    const { stream, handlers } = LangChainStream();

    //Initiate Pinecone Client
    const pineconeClient = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY ?? "", 
        environment: "gcp-starter",//environment: "us-east-1-aws",
    });
    /*const pineconeClient = new PineconeClient();
    await pineconeClient.init({
        apiKey: process.env.PINECONE_API_KEY ?? "", 
        environment: "us-east-1-aws",
    });*/

    const pineconeIndex: any = pineconeClient.Index(process.env.PINECONE_INDEX_NAME as string);

    //Initialize our vector store
    const vectorStore:any = await PineconeStore.fromExistingIndex(new OpenAIEmbeddings(),{ pineconeIndex });

    //Specify the OpenAI model we'd like to use, and turn on streaming
    const model = new OpenAI({
        modelName: "gpt-3.5-turbo",
        streaming: true,
        callbackManager: CallbackManager.fromHandlers(handlers),
        verbose: true,
    });

    //Define the Langchain chain
    const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever(), { //const chain = VectorDBQAChain.fromLLM(model, vectorStore, {
        //k: 1,
        returnSourceDocuments: true,
    });

    //Call our chain with the prompt given by the user
    chain.call({ query: body.prompt }).catch(console.error);

    //Return an output stream to the frontend
    return new StreamingTextResponse(stream);
}