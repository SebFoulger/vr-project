
files = {'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C18', 'C19', 'H01', 'H02', 'H03', 'H04', 'H07', 'H08', 'H13', 'H15'} 
for i = 1:16
    file = files{i}
    df=load(strcat(file, '.mat'));
    
    x = df.Trial.Trajectories.Labeled.Data(1,1,:);
    x(end) = [];
    y = df.Trial.Trajectories.Labeled.Data(1,2,:);
    y(end) = [];
    z = df.Trial.Trajectories.Labeled.Data(1,3,:);
    z(end) = [];
    
    vel_x = diff(x)*120/1000;
    vel_y = diff(y)*120/1000;
    vel_z = diff(z)*120/1000;
    
    speed = sqrt(vel_x.^2+vel_y.^2+vel_z.^2);
    
    time = (1:length(speed))/120;
    speed_clean = smooth(time, speed, 120 * 0.01, 'loess');
    
    writematrix(speed_clean, strcat(file, '.csv'));
end