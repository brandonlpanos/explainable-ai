; This program runs over all IRIS PF data producing SJI plots in each available filter
; The span of the raster is given as the space between two vertical lines, and the sampling time 
; corresponds to when these lines appear green
;------------- File structure -----------
; |--script
; |     --details
; |     --sji_movies.pro
; |--movies
; |     --observations
; |         --all SJI movies for obs
; |--data
; |     --obsevations
; |         --raster
; |         --sji

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
    read_iris_l2,rasterfiles[0],hdrs,dat

    ; iterate over different SJI's
    sji_data_paths = FILE_SEARCH(obs + '/sji/*')
    for ii=0,(size(sji_data_paths))[1]-1 do begin
        sji_path=sji_data_paths[ii]
        ; get string for filter
        splits = STRSPLIT(sji_path,'_',/EXTRACT)
        filter = splits[9] + '_' + splits[10]
        ; read in headers and data
        read_iris_l2,sji_path,sjihdrs,images
        images = IRIS_DUSTBUSTER(sjihdrs,images,bpaddress,clean_values,/run) ; clean dust from SJI
        times = sjihdrs.date_obs

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
            loadct,0
            tmpp = where(images[*,*,frame] eq -200)
            img = (images[*,*,frame]>0)^.3
            img[tmpp] = mean(img)
            index2map,sjihdrs[frame],img,map
            get_map_coord,map,xcoord,ycoord
            plot_map,map,ticklen=.02
            ; IRIS slit span
            slpos = xcoord[sjihdrs[frame].sltpx1ix-1,0] - sjihdrs[frame].pztx + hdrs.pztx
            clr='black'
            if date_conv(times[frame],'R') ge start_sample_time and date_conv(times[frame],'R') le end_sample_time then clr='green'
            plots,slpos[0],!y.crange,lines=0,color=cgcolor(clr),thick=0.5
            plots,slpos[n_steps-1],!y.crange,lines=0,color=cgcolor(clr),thick=0.5
            xyouts,.72,.80,filter,/norm,charsize=1.2, color=cgcolor('white')
            a=tvrd(/true)
            mymovie[*,*,*,frame] = a
        endfor

        ; compile movie
        video = idlffvideowrite('/Users/brandonlpanos/papers/AIA_IRIS/movies/'+obs_name+'/'+filter+'.mp4')
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