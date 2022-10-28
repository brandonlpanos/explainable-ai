; This program runs over all PF data producing AIA plots in each available filter
; The span of the raster IRIS raster is given as the space between two vertical lines, and the sampling time 
; corresponds to when these lines appear green
;------------- File structure -----------
; |--script
; |     --details
; |     --aia_movies.pro
; |--movies
; |     --observations
; |         --all AIA movies for obs
; |--data
; |     --obsevations
; |         --raster
; |         --sji
; |         --aia

; iterate over obsevations
details = rd_tfile('details.txt',3) ; text file containing details of all observations
n_obs = (size(FILE_SEARCH('/Users/brandonlpanos/papers/AIA_IRIS/data/*')))[1]
for i=0,n_obs-1 do begin

    ; load obs details, name, + stop, start times
    obs_details=details[*,i]
    obs_name=obs_details[0] ; for saving
    start_sample_time=date_conv((obs_details[1]),'R') ; convert to date_time to real numbers
    end_sample_time=date_conv((obs_details[2]),'R')
    obs='/Users/brandonlpanos/papers/AIA_IRIS/data/'+obs_name

    ; read raster details for ploting the rasters span
    path_to_raster_file = obs + '/raster/'
    rasterfiles = file_search(path_to_raster_file + '*raster*fits',count=nraster)
    read_iris_l2,rasterfiles[0],hdrs,dat ; only need to read in the first file for all the positions of the slit

    ; read SJI headers for IRIS's raster span
    sji_data_paths = FILE_SEARCH(obs + '/sji/*')
    sji_path=sji_data_paths[0] ; only need on of the SJI filters for the map and span
    read_iris_l2,sji_path,sjihdrs,sji_images
    iris_times = sjihdrs.date_obs

    ; turn list of IRIS's SJI times into real numbers to match position of slit to closest AIA images
    t_dim = size(iris_times)
    t_dim = t_dim[1]
    iris_times_r = List()
    for t=0, t_dim-1 do begin
      iris_times_r.Add, date_conv(iris_times[t],'R')
    endfor
    iris_times_r = iris_times_r.ToArray()

    ; iterate over different AIA filters
    dia_data_paths = FILE_SEARCH(obs + '/aia/*')
    for ii=0,(size(dia_data_paths))[1]-1 do begin
      aia_path=dia_data_paths[ii]
      ; get string for filter
      splits= STRSPLIT(aia_path,'_',/EXTRACT)
      wave = splits[9]
      filter = 'AIA_' + (STRSPLIT(wave,'.',/EXTRACT))[0]
      ; read in headers and data
      read_iris_l2,aia_path,aiahdrs,images
      aia_times = aiahdrs.date_obs

      ; set up dimensions
      shape = size(images)
      x_dim = shape[1]
      y_dim = shape[2]
      n_frames = shape[3]
      n_steps = size(dat)
      n_steps = n_steps[3]

      ; make movie
      set_plot,'X'
      device,RETAIN=2,SET_FONT='Helvetica Bold Italic', /TT_FONT
      window,xsize=500,ysize=500
      mymovie = bytarr(3,500,500,n_frames)
      siz = size(mymovie)
      for frame=0,n_frames-1 do begin
          !p.color=0
          !p.background=255
          !p.charsize = 1
          ; loadct,0, /silent
          aia_lct,rr,gg,bb,wavelnth=wave,/load
          img = (images[*,*,frame]>0)^.3
          index2map,aiahdrs[frame],img,map
          get_map_coord,map,xcoord,ycoord
          ; get map details for IRIS without ploting image
          time_of_aia_frame = date_conv(aia_times[frame],'R')
          near = min(abs(iris_times_r - time_of_aia_frame), index_of_closest_sji_frame)
          sji_im = sji_images[*,*,index_of_closest_sji_frame]
          index2map,sjihdrs[index_of_closest_sji_frame],sji_im,sjimap
          get_map_coord,sjimap,sji_xcoord,sji_ycoord
          ; IRIS slit span based on closest AIA and SJI frames
          slit_yarcsec = sjihdrs[index_of_closest_sji_frame].fovy
          slit_ybot = sjihdrs[index_of_closest_sji_frame].ycen - slit_yarcsec/2
          slit_ytop = sjihdrs[index_of_closest_sji_frame].ycen + slit_yarcsec/2
          ; IRIS FOV
          xrange_yarcsec = sjihdrs[index_of_closest_sji_frame].fovx
          xfov_left= sjihdrs[index_of_closest_sji_frame].xcen - xrange_yarcsec/2
          xfov_right= sjihdrs[index_of_closest_sji_frame].xcen + xrange_yarcsec/2
          plot_map,map,ticklen=.02,  xrange=[xfov_left,xfov_right], yrange=[slit_ybot,slit_ytop]
          ; plot slit span
          slpos = sji_xcoord[sjihdrs[index_of_closest_sji_frame].sltpx1ix-1,0] - sjihdrs[index_of_closest_sji_frame].pztx + hdrs.pztx
          clr='black'
          if date_conv(aia_times[frame],'R') ge start_sample_time and date_conv(aia_times[frame],'R') le end_sample_time then clr='green'
          plots,[slpos[0],slpos[0]],[slit_ybot,slit_ytop],lines=0,color=cgcolor(clr),thick=0.5
          plots,[slpos[n_steps-1],slpos[n_steps-1]],[slit_ybot,slit_ytop],lines=0,color=cgcolor(clr),thick=0.5
          a=tvrd(/true)
          mymovie[*,*,*,frame] = a
      endfor

      ; compile movie
      video = idlffvideowrite('/Users/brandonlpanos/aaa.mp4')
      framedims = [siz[2],siz[3]]
      framerate = 10.
      stream = video.addvideostream(framedims[0],framedims[1],framerate,BIT_RATE=9d8,codec='mpeg4')
      frame = bytarr(3,framedims[0],framedims[1])
      for j=0,siz[4]-1 do begin
          frame[0,*,*] = mymovie[0,*,*,j]
          frame[1,*,*] = mymovie[1,*,*,j]
          frame[2,*,*] = mymovie[2,*,*,j]
          timestamp = video.put(stream,frame)
      endfor
      video.cleanup
    endfor 
  endfor
END