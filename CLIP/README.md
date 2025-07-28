my own dummy expllanation of how in deep clip works , so the core idea is getting the embeddings from two seperate models and connectiing them in a joined space , what i will try to do is impelment how this joining work
concepts to go over
InfoNCE

resources 
https://medium.com/@paluchasz/understanding-openais-clip-model-6b52bade3fa3


during training , whqt is being updates are two weights matrixes ..wi of dims(n_batches,d)
and wt (n_batches,d)  with d ebing the dommun dimension we want to combine the text and image in , 


//the pytorch implementation of infonce:

