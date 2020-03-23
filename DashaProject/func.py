"""
   if len(img_tmp.shape) == 3:
       for i in range(0, img.shape[2]):
           img_y[:, :, i] = scp.convolve(img_tmp[:,:,i], dif_y_Gaus, mode = 'reflect')
           img_x[:, :, i] = scp.convolve(img_tmp[:,:,i], dif_x_Gaus, mode = 'reflect')
       img_x *= img_x
       img_y *= img_y
       for k in range(0, img.shape[2]):
           img_res[:,:,k] = np.sqrt(img_x + img_y)
   else:
   """

