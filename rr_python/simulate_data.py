

Just writing out pseudocode for now, but this will be the simulation exercise

random_seed = 1234

num_units = 200
num_periods = 20
days_per_period = 200
beta_low = 0.05
beta_high = -0.30
critical_threshold = 2
def true_gamma(alpha, sinusoid, detailed_return=False):
	# the production function, less idiosyncratic noise
	sinusoid = sinusoid[int(days_per_period / 4):int(days_per_period * 3/4)]
	low_pass = np.min(sinusoid, critical_threshold)
	high_pass = np.max(sinusoid, critical_threshold) - critical_threshold
	raw_yield = alpha + beta_low * np.sum(low_pass) + beta_high*np.sum(high_pass)
	if detailed_return:
		return raw_yield, np.sum(low_pass), np.sum(high_pass)
	else:
		return raw_yield

alpha_range = [1, 3]
sinusoid_amp_range = [1, 3] # Defines variability in "climate"
sinusoid_mean_range = [1, 3] # Defines mean temperature in "climate"
alpha_mean_corr = 0.5 # For a given location, how correlated are the climate + alpha
alpha_amp_corr = 0.5
sinusoid_var = 0.25
noise_var = 0.25 # Set variance of noise in yield equation 

location_characteristics = df 
full_panel = df
for i in range(num_units):
	alpha_i = uniform_draw(alpha_range)
	amp_i = uniform_draw(sinusoid_amp_range)
	amp_i = alpha_i * alpha_amp_corr + amp_i * np.sqrt(1 - alpha_amp_corr ** 2)
	mean_i = uniform_draw(sinusoid_mean_range)
	mean_i = alpha_i * alpha_mean_corr + mean_i * np.sqrt(1 - alpha_mean_corr ** 2)
	location_characteristics.append({'i': i,'alpha': alpha_i, 'amp':amp_i, 'mean': mean_i})
	for t in range(num_periods):
		sinusoid_mean = normal_draw(mean_i, sinusoid_var)
		sinusoid_amp = normal_draw(amp_i, sinusoid_var)
		sinusoid = sinusoid_mean - sinusoid_amp * np.cos(xrange(0, days_per_period) * 2 * math.pi / days_per_period)
		# sinusoid = sinusoid + normal_noise Not sure if I should add this step?
		raw_yield, ddays_below, ddays_above = true_gamma(alpha, sinusoid, detailed_return=True)
		epsilon = normal_draw(0, noise_var)
		noisy_yield = raw_yield + epsilon
		full_panel.append({'i':i, 't':t, 'sinusoid_mean':sinusoid_mean, 'sinusoid_amp':sinusoid_amp,
			'raw_yield':raw_yield, 'noisy_yield':noisy_yield, 'epsilon': epsilon, 'sinusoid':sinusoid, 'ddays_below':ddays_below,
			'ddays_above':ddays_above, 'ddays_below':ddays_below})
