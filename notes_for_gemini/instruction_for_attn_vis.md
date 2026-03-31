* Generate a code inside @examples/my_libero_sim to analyze save attention weights.
* We only analyze `step_attention`, which has shape (18,1,1,8,10,978).
  * 18 means 18 transformer layer.
  * 8 is the number of attention heads.
  * 10 is the action horizon/number of action tokens.
  * 978: the first 256*3 tokens are vision tokens from three images, then there are 200 text tokens.
* For plotting, we want:
  * One plot per epoch.
  * For each epoch, we have ten subplots: one for each action (our action horizon is 10).
  * For each subplot, x axis is `steps`, y axis is attention weight (should be 0~1).
  * In each subplot, we plot:
    * total attention weights from all visual tokens, averaged over 8 heads and 18 layers. 
    * total attention weights from all text tokens, averaged over 8 heads and 18 layers. 

  
