{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "%load_ext autoreload\n",
    "%autoreload 2\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Notebook to investigate the performance of spike interface in localizing neurons "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import MEArec as mr # what we will use to create a synthetic recording\n",
    "import spikeinterface.full as si  # what we will use to sort the spikes\n",
    "\n",
    "import warnings\n",
    "from probeinterface.plotting import plot_probe\n",
    "from matplotlib import cm\n",
    "from probeinterface import read_prb\n",
    "\n",
    "import time\n",
    "import numpy as np\n",
    "\n",
    "job_kwargs = {'n_jobs' : -1, 'chunk_memory' : '10M', 'verbose': True, 'progress_bar': True}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "rec = mr.load_recordings('recordings.h5')\n",
    "positions = np.hstack((rec.template_locations[:, 1:3], rec.template_locations[:, 0][:, np.newaxis]))\n",
    "\n",
    "from spikeinterface.sortingcomponents.benchmark.benchmark_peak_localization import BenchmarkPeakLocalization, plot_comparison_positions\n",
    "recording, gt_sorting = si.read_mearec('recordings.h5')\n",
    "recording_f = si.bandpass_filter(recording, dtype='float32')\n",
    "recording_f = si.common_reference(recording_f)\n",
    "recording_f = si.zscore(recording_f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "extract waveforms shared_memory multi buffer with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a43484a2f917427280ff8e50e0c519e8",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory multi buffer:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "extract waveforms shared_memory multi buffer with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "bb7097a424724328925018ba9b19e44b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory multi buffer:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "extract waveforms shared_memory multi buffer with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c26e09b6698f40e49043dbddbe4eb537",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory multi buffer:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "f281c48a0d644ee1a2854264cca4fc07",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "30c8d6563e244b07a5e2f321d6460a37",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3d0e05b12f8641aaa5d3ee6e7a821ad7",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "aeef1a1ed86d44a88f76d5c0a742aafd",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "fb643a7152834e5b944876d4defca5d9",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "95440882d4ef442e9a8247d9b427b692",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9904a1bbdb6043a29439f647106d9da4",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/media/cure/Secondary/pierre/softwares/spikeinterface/src/spikeinterface/sortingcomponents/tools.py:80: RuntimeWarning: invalid value encountered in divide\n",
      "  prototype = np.nanmedian(waveforms[:, :, 0] / (np.abs(waveforms[:, nbefore, 0][:, np.newaxis])), axis=0)\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8b4cd01d31904984a705d9f76a20b8f4",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using grid_convolution with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "f318153f60444ec78b1c438234721c5d",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using grid_convolution:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "76dddb85ac52401c83caf9567f386530",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4a83a04573dc463c9e811e0a85099563",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using grid_convolution with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b89b8c35071c45d28c82fb79292bbe4a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using grid_convolution:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using peak_channel with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a26e0f9838c84031a5c8dbfe76ea8823",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using peak_channel:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "benchmarks = []\n",
    "waveforms = None\n",
    "for method in ['center_of_mass', 'monopolar_triangulation', 'grid_convolution', 'peak_channel']:\n",
    "    if method == 'center_of_mass':\n",
    "        for feature in ['ptp', 'energy', 'peak_voltage']:\n",
    "            title = f'CoM ({feature})'\n",
    "            params = {'feature' : feature}\n",
    "            bench = BenchmarkPeakLocalization(recording_f, gt_sorting, positions, job_kwargs=job_kwargs, title=title)\n",
    "            if waveforms is not None:\n",
    "                bench.waveforms = waveforms\n",
    "            bench.run(method, params)\n",
    "            waveforms = bench.waveforms\n",
    "            benchmarks.append(bench)\n",
    "    elif method == 'monopolar_triangulation':\n",
    "        for feature in ['ptp', 'energy', 'peak_voltage']:\n",
    "            title = f'Monopolar ({feature})'\n",
    "            params = {'enforce_decrease': True, 'feature' : feature}\n",
    "            bench = BenchmarkPeakLocalization(recording_f, gt_sorting, positions, job_kwargs=job_kwargs, title=title)\n",
    "            if waveforms is not None:\n",
    "                bench.waveforms = waveforms\n",
    "            bench.run(method, params)\n",
    "            waveforms = bench.waveforms\n",
    "            benchmarks.append(bench)\n",
    "    elif method == 'grid_convolution':\n",
    "        for feature in ['gaussian_2d', 'exponential_3d']:\n",
    "            title = f'Grid ({feature})'\n",
    "            params = {'weight_method': {'mode' : feature}}\n",
    "            bench = BenchmarkPeakLocalization(recording_f, gt_sorting, positions, job_kwargs=job_kwargs, title=title)\n",
    "            if waveforms is not None:\n",
    "                bench.waveforms = waveforms\n",
    "            bench.run(method, params)\n",
    "            waveforms = bench.waveforms\n",
    "            benchmarks.append(bench)\n",
    "    elif method == 'peak_channel':\n",
    "        title = 'Peak Channel'\n",
    "        params = {}\n",
    "        bench = BenchmarkPeakLocalization(recording_f, gt_sorting, positions, job_kwargs=job_kwargs, title=title)\n",
    "        if waveforms is not None:\n",
    "            bench.waveforms = waveforms\n",
    "        bench.run(method, params)\n",
    "        waveforms = bench.waveforms\n",
    "        benchmarks.append(bench)\n",
    "    \n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "GridSpec(4, 3)\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1080x720 with 8 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "colors = {}\n",
    "\n",
    "import matplotlib as mpl\n",
    "\n",
    "cmap = mpl.cm.tab20c\n",
    "norm = mpl.colors.Normalize(vmin=0, vmax=1)\n",
    "\n",
    "for value, bench in zip([0,0.05,0.1, 0.4,0.45,0.5, 0.6, 0.65, 0.9], benchmarks):    \n",
    "    colors[bench.title] = mpl.cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(value)\n",
    "    \n",
    "plot_comparison_positions(benchmarks, colors=colors)\n",
    "import pylab as plt\n",
    "plt.savefig('comparison.pdf')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "extract waveforms shared_memory multi buffer with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4b4ff6289f104588849784f2fe0ff8da",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory multi buffer:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "extract waveforms shared_memory multi buffer with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "cacc9d1909b74306867da4158f2b3767",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory multi buffer:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "extract waveforms shared_memory multi buffer with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "42770c69e1ad49baba76c5729fcd6e88",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory multi buffer:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/media/cure/Secondary/pierre/softwares/spikeinterface/src/spikeinterface/sortingcomponents/benchmark/benchmark_peak_localization.py:583: UserWarning: *c* argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with *x* & *y*.  Please use the *color* keyword-argument or provide a 2D array with a single row if you intend to specify the same RGB or RGBA value for all points.\n",
      "  axs[0, 2].scatter(b.template_positions[cell_ind, 0], b.template_positions[cell_ind, 1], c=colors[b.title], s=100)\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA5IAAAKoCAYAAAAF58BtAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy81sbWrAAAACXBIWXMAAAsTAAALEwEAmpwYAAEAAElEQVR4nOzdd3xb1fn48c+xLUuyZctWHDsJGc7ewxBWQgptQpkNtFBoKVD6bYFSun4tdDIKdA9KS1soLV1AGW2BUmZJKWlIWAmOTTZO4mzbcSzLlizLQ/f3x7FsWda0JUt2nvfr5Zelq3vuOfdKce6j5wxlGAZCCCGEEEIIIUS8stLdACGEEEIIIYQQI4sEkkIIIYQQQgghEiKBpBBCCCGEEEKIhEggKYQQQgghhBAiIRJICiGEEEIIIYRIiASSQgghhBBCCCESkpPuBgghxPFo06ZN5qysrLsMw7jeMIwCQKW7TUKIITGysrJ2+P3+D5500kkH090YIYRINQkkhRAiDbKzs9fk5+efOHny5Lzc3FyUkjhSiJHM7/erI0eOzN6/f/8bq1evXvjMM884090mIYRIJenaKoQQadDd3b1s+vTpeWazWYJIIUaBrKwsxo8fn2UymU4AvrZ69eridLdJCCFSSQJJIYRIj6ysLPkTLMRokpWVFfhiaDywIs3NEUKIlJK7GCGEEEKI5PKgg0khhBi1JJAUQgghhEguI90NEEKIVJNAUgghRNzuuecePv/5zwNw00038cMf/rD3NZ/Px/XXX8/MmTMpKChg8uTJ3HzzzbS3t/c7xk9+8hNOOOEE8vPzWbVqFXv27BnWcxDHF/nMCiFEakggKYQQIm5vvvkmp512GgCvv/5672OArq4uSkpK+Ne//kVzczPr1q3jlVde4Wtf+1rvPo888gg/+clP+Ne//sXRo0eZN28eq1evpru7e9jPJdN1dnamuwmjgnxmhRAiNSSQFEIIEbc33niDU089lY6ODqqqqli6dGnva/n5+Xzve99jzpw5ZGdnM2XKFK699lpeffXV3n0eeOABrr/+ek488UTy8vL4/ve/z549e3jttdfScDaZpby8nDvvvJP3v//92Gw2Hn/8cb7//e8za9YsioqKWL58ORs3buzd3zAMHnjgARYuXEhhYSGTJk3iV7/6VRrPIDPJZ1YIIVJD1pEUQohMMdzLgBjxDeNau3YtF110EYZh0Nraysknn0x3dzft7e1MnDiRE044ga1bt4Yt+5///IfFixf3Pq+qquL//b//1/vcZrMxc+ZMqqqqOPPMM4d2PnH48pe/zObNm1NeD8CSJUu45557Eirzu9/9jmeeeYYlS5Zw00038dprr/Hiiy8yZcoU/vSnP3Huuefy3nvvUVxczP33389dd93FE088wbJly2hqamLv3r2pOZkwNg3z5/WkOD+vMLo+s0IIkakkkBRCCBHVmWeeSXNzM3/+85954YUXeOyxx7jzzjtxu938+Mc/jljunnvuYe3atf2yaK2trdjt9n77FRUV0dLSkrL2jyTXXnstFRUVvdnG5557jmnTpgHw6U9/mnvuuYfnnnuOK6+8knvvvZdvf/vbnHHGGQCUlJRQUlKSzuZnDPnMCiFE6kkgKYQQmSKBjEs6vPLKK7z//e8H4NVXX+Wmm26KuO/Pf/5zfvSjH/HKK68wefLk3u0FBQW4XK5++zY3N1NYWJiaRodINEM43MrLywFobGzE7XbzoQ99KLAuIaDHTR48eBCA2tpaZs2alY5mAollCNNlNHxmhRAiU0kgKYQQIqqioiIAWlpaeOqpp/ja175Ga2srmzZtQinF66+/zty5c3v3v+uuu/jtb3/L2rVrmT17dr9jLV68mHfeeYeLL74YALfbzXvvvdevK+HxLCtLT11QUlJCfn4+a9as4eSTTw67b3l5Oe+99x5nn332cDZxRJDPrBBCpJ5MtiOEECKq5uZmXnvtNWbPnk1LSwuPPPIIq1atwuVy0dzc3O+G/Oabb+b3v/992BtygOuuu47f/va3VFZW4vV6ueWWW5g6dWpv90yhKaX40pe+xE033cR7770H6ADmpZde4vDhwwDceOONfP/73+f111/H7/fT2NjI22+/nc5mZwz5zAohROpJRlIIIURM//jHP/jIRz7S+/jSSy8dsM++ffv46U9/Sm5ubr9szZQpU3onNvnEJz7BoUOHuOCCC2hubub000/nmWeeITs7e3hOZAS54447+OUvf8lFF13EwYMHyc/P57TTTuPee+8F4HOf+xygx07u378fh8PBN77xjYgZzOONfGaFECK1lDECxjgIIcRos2nTJuOkk05KdzOEEEm2adMm7rjjjl8D7z7zzDO/TXd7hBAiVaRrqxBCCCGEEEKIhEggKYQQQgghhBAi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5/cbmxoajKdSEARBOOuROpJ9sGTJEr1ly5YT6nv48GGmTJlyii0SBOFk0VqfDZkR5Uv7RNi9G955xwigq6823rNt22DrVpjcuafL64WSEli0yHjhjh0zQs7tNtlIDx/u2gO4eLERZZs3GxH15S8PbMPWraZPWZl5LioydmzdagRkaanZf1hUBAsWwKFD8OKLxvsXCBjRt2wZXHihuZ7t240gdThMnwsuGNw+ySefNEIxM9OIRp8P9u83QraiwnhqCwuNjZMnG2/ohg3G7tRU4xGdN68ry+3hw8YjGQ7DqlXGRmGonA3fTYwrydGyR3JgZI/kwMgeycFxevZIbt6qtV7S2znxSA4DkydP5mtf+xp//vOfGTt2LBdddBF5eXls3ryZhoYGsrOzaWxsZMWKFbz00ktccsklhMNhnnvuOWpqali0aBErVqzg2WefpaKigtzcXK677jpsNht+v59AIMCRI0eYN28ejz32GOeffz65ubmMHz+e1NRU9u7dS2trKwDz589n7dq1tLa2UlZWBsC9995LY2Mj//3f/w3AtGnT8Hq9BAIBGhoamDZtGuPHj2fatGlkZmbS2NjII488wjXXXAOYjLZFRUU8/fTTjBs3juuvv560tDTeeustNm7cyKc+9SkqKysJhULk5OTw0ksvcffdd7NhwwbWrFnDypUrqaysZNasWaSkpPDuu+/GbbvllltQSnHJJZfw9ttv86c//YmUlBSmTJmCx+NhwYIFtLe3s3fvXlwuF5deeiljxozhiSeeID8/n0OHDjFjxgyuuOIK3n33Xd58801mzJjBww8/zOOPP86ePXt455134uvw3nvv0djYyLp16/iv//ov7HY7mzdvZseOHezdu5c5c+Zw3XXX4ff7eeaZZ1i8eDFWq5WqqipsNhsrVqygpqYGgD/84Q8AfOUrX+G8887jqaeeYu3atSQnJ3PzzTdTWlrKG2+8wfTp0/H5fNTX1/ORj3wEl8vF/fff3+0zdMstt5CcnMysWbM4cuQIW7duZf369QDMnDmTuXPnsnnzZoqLi2lsbOS2224jNTWVnJwc8vPzsVgsaK2xJGRxDAaDuFwu/H4/4XAYm80W9xoDaK2x2WyEQib7l9PpxOVyEY1GCQaDOByOeD+ASCSCz+eLj+NwOEhPT6e6uppQKERycjJjxowhGAzG1wjAarWSlJQU99gq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nD/+sfE0nzmcDd9NskdSEM5C+tsjKR7JYWDKlCncdtttPProo+Tm5vKRj3wEn8/HoUOH2LRpE5/+9Kf561//SltnFrq//vWvHDhwgI0bN/L8889jtVrJzMxk7ty52Gw2lFKUlZVx6NAhrr76ag4fPtyttMV5551HIBAgOTmZqVOn8oc//IH09HQ+//nP89577/H22293s2/atGnccccd8Uyju3fvxmq1kpGRQXNnnbKkpCQyMjJYsWIF9fX1NDc3EwgEmDZtGjk5ObzyyitEo1FKSkpYvXo1tbW1PPTQQ93m+fGPf4zdbueBBx5Aa82//uu/8sADD7BgwQLKy8u5/PLL+c1vfoOvM+uiy+XiG9/4BsXFxTz33HM899xzACxYsIDvfve7lJeXs2fPHkpLS9m6dSs33ngjf/jDH7jmmmtIT08nPT2d1tZWkpOT+cMf/kBKSgoejweAF154gcrKSr71rW/R3NzMHXfcwf79+3nnnXdIT08nGAzywAMP8PLLL/P3v/+dK664gvPOO4+XXnqJ7du3x6+poKCA2s69QNdeey0zZ84kNzeXd999l5deeolPfepTzJ07F5fLxbp163jhhRdYvXo1//jHPwgGg1x88cXU1NQwZswYampqWLFiBcXFxdx3330kJSVxySWXoLVmy5YtVFdX09jYyHXXXceLL77IlClT+PjHP8769evZvHkzl156KbW1tWzZsoVFixZxzTXX8Mgjj1BcXExubi6vvfYawWAQpRR33nknzz//PAUFBYRCIVasWMH69etZv359vN7jypUrKSws5IknnqC9vR2tNTNmzGDbtm2cd9551NfXs3r1apYvX87KlSvja/L222/zpz/9iRdeeAGbzcaxY8cA+MlPfsLjjz/OvHnzeOaZZ+jo6OALX/gCZWVlvPzyywCMHTuWK664gpqaGnJycnjiiScAKC4upqKigpKSEo4ePQrArFmzWL58eTwz7rvvvgvA8uXL8fv9XHnllaxfv5758+cTDoeprq7m+eef58tf/jIvvvgiR44cOZF/zsLJ8uyzJkFLUpLx4iWWlQiFjCcwNdWIq7o6I9Ri2UF9vq4spvX1RkCC6ff228aLNX36wEIylgG0rMyM39ZmBEwo1H3vpMViRIzdbp537jSiyuczAs/rNcJGayOqlDJCMpYoZyC2bIGNG7v2f9psRtw6ncaL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ccRxzhDfI3kCnM4eyUOHDkUCnpwOtAAr8TE6dWRmZtLb23u+zTgvUBQlPndGIR61dQy0twuZyM098TUDA6IQafvhTqUOLRfkichJe7u85+cLyTh8WIhiWVn0vMkkZWiRVBMTR+Z3PBmCQdljqKqy39FgiNZpMEiZiYmyv669Pbof0+mUV3a27AE0GsWOU9mbCLLfLz1dysrNje7zS0oSYms0St2KImRK23N45IiMU3KyEF+vV9p+quTebpc9mbF7GmP3SIIEuhkakvE41bE/Ebq7pfzERJlber30gc0mbRscjBLJ/n4h1uFwlMD6/XJtY6P0R0XFyH22ZwtnT/E8p79NiqKYgV8CVwDpwBHgy6qqvjB8/nLgF8AUYAtwl6qqx8YrN75HMo44LjzE90ieY5SXl/OJT3yC//f//h8Al112GYsXL+af//wn9fX1ACxdujQSTbSoqIi//OUvkYTrv/vd7+js7GTTpk1kZGRw00038be//Y1nnnkGk8nEkiVLmDJlCk6nk7q6Og4dOkRFRQW5ubnccMMNvPLKK/znP/8B4L777uP222+noaGBj3zkIwB86lOf4ktf+hLvf//7efXVVykqKqKrqwv/sPuSyWTiuuuuY+fOnRw7dozc3Fw6Ozu55557+P3vfw/IXseVK1diMBhobm4mIyODZ555hsWLF3P11Vezbds2GhoacDqddA1H4PvUpz7FP//5TwYGBsjMzOR973sfBw8eZNu2baSnp7Nr1y7mzZvH5ZdfTlVVFffffz9er5fKyspIkvusrCxuvfVWenp62LZtGyaTiaSkJHbs2MGqVas4duwYH/jAB7j55pu59dZbqa2tpaioiN///vfs2rWLZ555hs2bNwNQXFzMLbfcgt/v5+GHH2bWrFkcOnRoROL77Oxspk2bxq5duxgaGuLmm28mEAjw/PPPY7PZsNvtJCYmRqLcavdcffXVZGRksGvXLpKTk2lpaaGjowOdTkdlZSVlZWXs2rWLuXPnEgwG+ctf/kJeXh42m41LL70Ur9dLX18f5eXlhEIh9uzZwxtvvAHAhz70ITweD+vXr2fOnDl0dnayY8cOAK6//no6Ojq4/PLLaW5uxufzodfrOXDgALW1tYBEzq2pqeHFF1+M9MPq1av529/+hsPhYPHixaxcuRKj0cj3v/99LBYLF198MZWVlfzv//7viLle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\n",
      "text/plain": [
       "<Figure size 1080x720 with 6 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "from spikeinterface.sortingcomponents.benchmark.benchmark_peak_localization import BenchmarkPeakLocalization, plot_figure_1\n",
    "plot_figure_1(benchmarks[::3] + [benchmarks[-1]], colors=colors)\n",
    "plt.savefig('illustration.pdf')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "from spikeinterface.sortingcomponents.benchmark.benchmark_peak_localization import plot_comparison_precision"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "extract waveforms shared_memory multi buffer with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c9ef96d9e26c46d88c59b96285d0eb4d",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory multi buffer:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "extract waveforms shared_memory multi buffer with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "bdc8edc78bc143c6937fa56a1b18ea24",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory multi buffer:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "extract waveforms shared_memory multi buffer with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a85ab14c833e44b0be69a77bb41b9b80",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory multi buffer:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "2ff2958cb115431ba3c52e7acb342240",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "22a2988e49184646a409c207c6abfc03",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4158a4faad40473c9fcfe836bb7c449b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "96914e30926047219cd85eabf6ef8162",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "fc655360e2144059a6c72d8fd9f787ff",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9844c4c3698f454fa0862ed9efad245b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c23aaa6a93c641878aea71e871de6629",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "35ab48fd86624e58a271897b17d6cdb0",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "0611bb39bb9844a7bbb320d2b27fabaa",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "02c010146fe64afba9087d17a52fbdad",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c786c460070642c682cacc8830a1171e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e64040a2132f498b85f2c31bf24f0e4d",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ff5ccbb3155d40f48868de13c2684a18",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "fec3e45722aa43319b6dded596f94ee5",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "dc4e5994c5314ff48b51b9c37cbe4849",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e0b2e9ebe5534ed8a043baf9075fa2c3",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "70ca4bc31ca9406a88e6fc735f08327d",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "be67574eeb2e4ed489b6ccd20e291a92",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8c48d80771614bb1b6400e805ecbe5ac",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4b65e8f44aa7460983963a64ad06c845",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "7bc53612306441d682f8338b7c0d1ade",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9ec91d69a3484495b106369a3af608be",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "656bc65f756d4e39a64097492b6799df",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "435fb4d9946a4f8980b0d867423ccb3e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6990bdc1beef47ccbf57f4f9ec817c24",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "49df70ee51b248848814e1f68016e3c2",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "bbcf7886469f4e94bfbfe5f329c077ff",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9e0736b5326f4cf784fcefa29f51f4f1",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "acd612282f224101b59aea93526a56b2",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6ca4a3285a514d8f97be688d2b2997c4",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c86f04a3546e494699a0dac2989af2c9",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3b70a68d52454bef9763a6fba4d6f367",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9064b5a9aded4fe0b4adcb5d6d2abc02",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4ef8a9c06e2d4e508d803b44d2f55667",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "56c8a12d907b4d98b05f42abf9d31deb",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d93838dfa266483ca708b7b8acdd18f0",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b67622648818431a85ed9868dadb016e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/media/cure/Secondary/pierre/softwares/spikeinterface/src/spikeinterface/sortingcomponents/tools.py:80: RuntimeWarning: invalid value encountered in divide\n",
      "  prototype = np.nanmedian(waveforms[:, :, 0] / (np.abs(waveforms[:, nbefore, 0][:, np.newaxis])), axis=0)\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b5cdb927a0b54a358c55cd988409fc28",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using grid_convolution with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "423f8b94c2a145049e94b9f619bdc16e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using grid_convolution:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e11d3c54f028422682fc78fa13c837da",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/media/cure/Secondary/pierre/softwares/spikeinterface/src/spikeinterface/sortingcomponents/tools.py:80: RuntimeWarning: invalid value encountered in divide\n",
      "  prototype = np.nanmedian(waveforms[:, :, 0] / (np.abs(waveforms[:, nbefore, 0][:, np.newaxis])), axis=0)\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "177d7449f0474167a1dee045560a5395",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using grid_convolution with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d092322cf09c4146b881f1ef6a2baf5a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using grid_convolution:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "87eb81a3f8a947e88139ee5198efac5c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5c02f8a7510d4c5483eba1b9ac39e343",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/media/cure/Secondary/pierre/softwares/spikeinterface/src/spikeinterface/sortingcomponents/tools.py:80: RuntimeWarning: invalid value encountered in divide\n",
      "  prototype = np.nanmedian(waveforms[:, :, 0] / (np.abs(waveforms[:, nbefore, 0][:, np.newaxis])), axis=0)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using grid_convolution with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "02bb941ad19e4fdf96bbd1330db16ec1",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using grid_convolution:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b186adc40de146bdaf4cab556dd6b3ea",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e30b16003ed04a89b4e012a68b5a15a3",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using grid_convolution with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6a7a16a66056445fb59dde8a6cdb279f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using grid_convolution:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "97aa460311304e36aa7595228996df0b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9d65a4a7f472488a90f3fb5e7926fdac",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using grid_convolution with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "493c30208af041cf88e104975dd7263d",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using grid_convolution:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "785f42e98001465499dbf9dbeb4c0216",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6bd76eabf78444cfb796e89d712345bb",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using grid_convolution with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "7f9c9e20f5fb40db8ec0f19de5975858",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using grid_convolution:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "dfd7c70e61ee46e7b28395eaacd7c477",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ca47f660d30d4a12b0fe66bc8e2b2d1c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using grid_convolution with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c92cc6da8877478b9d8c68eece3f0df1",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using grid_convolution:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "7bd1c31a69734a71bc280c9ba30ebb6b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ae2354474988499dbf0ad55a67f46c4f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using grid_convolution with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "149f5617e8ee451fb511589c5b1be5f2",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using grid_convolution:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "11d6af5d18ab4078b7570f549465b2ed",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "0fbcbca4ddd149709b48dd2b4eab8258",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using grid_convolution with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3e18e68ee86640548e674bb8624ea84f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using grid_convolution:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "bd83d575aa0f43afb64328b599314d8a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "65017fcea4004554bff76ed8f30db71e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using grid_convolution with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "836267050ef54153816e1601b9f4401f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using grid_convolution:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d10f98ba10b840a68447adcc22fb98ad",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b857413d255c4945817051630d518f1d",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using grid_convolution with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a875629d25e34af5ad6ffdc105585d84",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using grid_convolution:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "79de74fe6c09442ea0f4545c9bba65c7",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/media/cure/Secondary/pierre/softwares/spikeinterface/src/spikeinterface/sortingcomponents/tools.py:80: RuntimeWarning: invalid value encountered in divide\n",
      "  prototype = np.nanmedian(waveforms[:, :, 0] / (np.abs(waveforms[:, nbefore, 0][:, np.newaxis])), axis=0)\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "88880abe2f9f4f0fa4ad054eaf54364e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using grid_convolution with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1bed19cbe61342c1be37906ec147ccf8",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using grid_convolution:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "benchmarks_ms = {}\n",
    "waveforms = None\n",
    "xaxis = [0.25, 0.5, 0.75, 1, 1.25, 1.5]\n",
    "benchmarks_ms['xaxis'] = xaxis\n",
    "\n",
    "for method in ['monopolar_triangulation', 'center_of_mass', 'grid_convolution']:\n",
    "    for t_range in xaxis:\n",
    "        if method == 'monopolar_triangulation':\n",
    "            for feature in ['ptp', 'energy', 'peak_voltage']:\n",
    "                title = f'Monopolar ({feature})'\n",
    "                params = {'enforce_decrease': True, 'ms_before' : t_range, 'ms_after' : t_range, 'feature' : feature}\n",
    "                bench = BenchmarkPeakLocalization(recording_f, gt_sorting, positions, job_kwargs=job_kwargs, title=title)\n",
    "                if waveforms is not None:\n",
    "                    bench.waveforms = waveforms    \n",
    "                bench.run(method, params)\n",
    "                waveforms = bench.waveforms\n",
    "                if title in benchmarks_ms:\n",
    "                    benchmarks_ms[title] += [bench]\n",
    "                else:\n",
    "                    benchmarks_ms[title] = [bench]\n",
    "        elif method == 'center_of_mass':\n",
    "            for feature in ['ptp', 'energy', 'peak_voltage']:\n",
    "                params = {'ms_before' : t_range, 'ms_after' : t_range, 'feature' : feature}\n",
    "                title = f'CoM ({feature})'\n",
    "                bench = BenchmarkPeakLocalization(recording_f, gt_sorting, positions, job_kwargs=job_kwargs, title=title)\n",
    "                if waveforms is not None:\n",
    "                    bench.waveforms = waveforms    \n",
    "                bench.run(method, params)\n",
    "                waveforms = bench.waveforms\n",
    "                if title in benchmarks_ms:\n",
    "                    benchmarks_ms[title] += [bench]\n",
    "                else:\n",
    "                    benchmarks_ms[title] = [bench]\n",
    "        elif method == 'grid_convolution':\n",
    "            for feature in ['gaussian_2d', 'exponential_3d']:\n",
    "                title = f'Grid ({feature})'\n",
    "                params = {'weight_method': {'mode' : feature}, 'ms_before' : t_range, 'ms_after' : t_range}\n",
    "                bench = BenchmarkPeakLocalization(recording_f, gt_sorting, positions, job_kwargs=job_kwargs, title=title)\n",
    "                if waveforms is not None:\n",
    "                    bench.waveforms = waveforms\n",
    "                bench.run(method, params)\n",
    "                waveforms = bench.waveforms\n",
    "                if title in benchmarks_ms:\n",
    "                    benchmarks_ms[title] += [bench]\n",
    "                else:\n",
    "                    benchmarks_ms[title] = [bench]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "extract waveforms shared_memory multi buffer with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b77a771123ca4648996d7bb463810fa9",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory multi buffer:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "extract waveforms shared_memory multi buffer with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "62f9cc4c11254032b2106196c1fe227c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory multi buffer:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "extract waveforms shared_memory multi buffer with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e604d88b59f242078f7217231228d46a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory multi buffer:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "208f3c1dfb454592a78096ea52e31420",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "aed5a64a22fe4ffa85cf0fa8543377fc",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5de91119925f41819d1b0c07c6c75c4f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "26513641f611439ca85cb32477f7b006",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a4c6470bf2574e01a9bb1c92896eafd0",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5f48ba706b41459c883418e1befbe578",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "242d3e22455a4d9d82b6f22fcda10024",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "f29e6e36b6254329af1575e9a7916299",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8377358b6bad46d19281c7e2abab7638",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3029cb019f6640f78ca2dfe5017a4950",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8c11ed323f9647b4bfbb2dc87a2c441f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "04857534cf1241588a668a287acc0977",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "cc537b59f0674fad8baa1237f350d5b1",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "de92ea5e172e4f118269ba2bd6a4d05d",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1cf4126cabad42d1a59142519cef0056",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9a3b9f488aa64814b55edc736ba57f0b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "37a3476c4f564a11b3f27ab05ff662f0",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4278bd904d0a4be886295e8b473d6566",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "0ff4a7238e49478baea04cf8da2e6e8a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a9dba658f6d740e1922baec3173967a3",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "cae7608724454f6b9c7c155e78b01c1d",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4e35e5bfaf6c4aefa6fe0a41d5bb11a7",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "49648b503b8442b38761a6b14cda1b6c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "06362d1a00e541ddaa5a2421a8ff4253",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "813217d9cf3746b8aa461e872a95daf4",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "0c9ee3b0bfbe41bdad3d279e6da150aa",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8065a93e2fad4b1f8afccca50c49a7b9",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "68e6e89ea7804e6b8bd10306b4c11d97",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "15c43ddc00af4a5f8db69d0fab2cd8af",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c0f160d58b704c24babe335e0ffb5cdf",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6ac314d7118146e9b383e9690614f173",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e83aa5bdc35a488bbec453e8ce23c633",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1b7c01f94c6147c1903470661876d40d",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "452fed844d914d9d8dcd1c6593109938",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "074792d837124820a649c42492c4b77d",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "fede767a8edd4c84a7d6c2e2a4bc59af",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "03ceab3e1a81405c8b71f70fa9b3e549",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/media/cure/Secondary/pierre/softwares/spikeinterface/src/spikeinterface/sortingcomponents/tools.py:80: RuntimeWarning: invalid value encountered in divide\n",
      "  prototype = np.nanmedian(waveforms[:, :, 0] / (np.abs(waveforms[:, nbefore, 0][:, np.newaxis])), axis=0)\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "df9f6bcc8b604b15b8f5b4df14e83791",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using grid_convolution with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "f430ca03b09c427f8de1cb187826ef55",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using grid_convolution:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "079086430ac0440b8a8cb437f5b24f1d",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/media/cure/Secondary/pierre/softwares/spikeinterface/src/spikeinterface/sortingcomponents/tools.py:80: RuntimeWarning: invalid value encountered in divide\n",
      "  prototype = np.nanmedian(waveforms[:, :, 0] / (np.abs(waveforms[:, nbefore, 0][:, np.newaxis])), axis=0)\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a15f191c601247008474a8b8170d2570",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using grid_convolution with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "aed3bd219ecd4794ad3277b5437aee8e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using grid_convolution:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "89a0ab46c5034499ad525a72c0ba633b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "bdcf9c9270194328a86e968a9f6ceafe",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using grid_convolution with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "68b48851fc77454da3c85a13116e732e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using grid_convolution:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b2d369a079c2446eabbc4cd93a427538",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "cd497227e0d14bd194ee175482ba07f7",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using grid_convolution with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "bbebf7af81c54bd1bd73a314ec54110e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using grid_convolution:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "79e94a1757cf42bfb330f7f867a701d5",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "deb9236d2668433a8a73134ceb82e805",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using grid_convolution with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "51e8d09a84ee4e3abda0ae3ce43ad3f7",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using grid_convolution:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "29cc4c9e2f1a4b99a6634b35e518f648",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/media/cure/Secondary/pierre/softwares/spikeinterface/src/spikeinterface/sortingcomponents/tools.py:80: RuntimeWarning: invalid value encountered in divide\n",
      "  prototype = np.nanmedian(waveforms[:, :, 0] / (np.abs(waveforms[:, nbefore, 0][:, np.newaxis])), axis=0)\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1800191d5200463e808b686e96dc42f1",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using grid_convolution with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e0d6c56bce9d4e599a1be655b31e8762",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using grid_convolution:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "893ddfc9d2be4fc8af33eccea4924ee2",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6db20df866464210bb7c0c3c7888882c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using grid_convolution with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9e7ac85952434d60b379f9221b6a443f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using grid_convolution:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "aded816a584d48d384aca6d5251fd774",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9b208efd94eb431aba3ef28d9790882f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using grid_convolution with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e90b7a586c3f4d93819584672cd32d55",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using grid_convolution:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "42f5ebb2ce2842f9bdb18f3bb4031276",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/media/cure/Secondary/pierre/softwares/spikeinterface/src/spikeinterface/sortingcomponents/tools.py:80: RuntimeWarning: invalid value encountered in divide\n",
      "  prototype = np.nanmedian(waveforms[:, :, 0] / (np.abs(waveforms[:, nbefore, 0][:, np.newaxis])), axis=0)\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9872a9a6454b49fba2217d82a0fd5e68",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using grid_convolution with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ee8dad83500f492ea50741522db7dec3",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using grid_convolution:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b09164995cf543a5816e8ae90701268c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/media/cure/Secondary/pierre/softwares/spikeinterface/src/spikeinterface/sortingcomponents/tools.py:80: RuntimeWarning: invalid value encountered in divide\n",
      "  prototype = np.nanmedian(waveforms[:, :, 0] / (np.abs(waveforms[:, nbefore, 0][:, np.newaxis])), axis=0)\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "62ca5dba45be4bbfaac37ef71ee033d1",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using grid_convolution with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "0c01334fecf347e889e13d50ef40ebe4",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using grid_convolution:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "aa6082f4fdf44923afbb454c8b8ca424",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d87fcda3ba7246e1914b6d27a337e254",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using grid_convolution with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e1d8c7e0791141b9aabaeceac9976d09",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using grid_convolution:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "0bf3d22442684675b396d8ff517b9fab",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/media/cure/Secondary/pierre/softwares/spikeinterface/src/spikeinterface/sortingcomponents/tools.py:80: RuntimeWarning: invalid value encountered in divide\n",
      "  prototype = np.nanmedian(waveforms[:, :, 0] / (np.abs(waveforms[:, nbefore, 0][:, np.newaxis])), axis=0)\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "0bb513fdc156423a85aadb30d6671c5b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using grid_convolution with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "86e2e1e223214aeca2b6d5c3958e1a2b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using grid_convolution:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "benchmarks_radius = {}\n",
    "waveforms = None\n",
    "xaxis = [25, 50, 75, 100, 125, 150]\n",
    "benchmarks_radius['xaxis'] = xaxis \n",
    "\n",
    "for method in ['monopolar_triangulation', 'center_of_mass', 'grid_convolution']:\n",
    "    for r_range in xaxis:\n",
    "        if method == 'monopolar_triangulation':\n",
    "            for feature in ['ptp', 'energy', 'peak_voltage']:\n",
    "                title = f'Monopolar ({feature})'\n",
    "                params = {'enforce_decrease': True, 'radius_um' : r_range, 'feature' : feature}\n",
    "                bench = BenchmarkPeakLocalization(recording_f, gt_sorting, positions, job_kwargs=job_kwargs, title=title)\n",
    "                if waveforms is not None:\n",
    "                    bench.waveforms = waveforms    \n",
    "                bench.run(method, params)\n",
    "                waveforms = bench.waveforms\n",
    "                if title in benchmarks_radius:\n",
    "                    benchmarks_radius[title] += [bench]\n",
    "                else:\n",
    "                    benchmarks_radius[title] = [bench]\n",
    "        elif method == 'center_of_mass':\n",
    "            for feature in ['ptp', 'energy', 'peak_voltage']:\n",
    "                params = {'radius_um' : r_range, 'feature' : feature}\n",
    "                title = f'CoM ({feature})'\n",
    "                bench = BenchmarkPeakLocalization(recording_f, gt_sorting, positions, job_kwargs=job_kwargs, title=title)\n",
    "                if waveforms is not None:\n",
    "                    bench.waveforms = waveforms    \n",
    "                bench.run(method, params)\n",
    "                waveforms = bench.waveforms\n",
    "                if title in benchmarks_radius:\n",
    "                    benchmarks_radius[title] += [bench]\n",
    "                else:\n",
    "                    benchmarks_radius[title] = [bench]\n",
    "        elif method == 'grid_convolution':\n",
    "            for feature in ['gaussian_2d', 'exponential_3d']:\n",
    "                title = f'Grid ({feature})'\n",
    "                params = {'weight_method': {'mode' : feature}, 'radius_um' : r_range}\n",
    "                bench = BenchmarkPeakLocalization(recording_f, gt_sorting, positions, job_kwargs=job_kwargs, title=title)\n",
    "                if waveforms is not None:\n",
    "                    bench.waveforms = waveforms    \n",
    "                bench.run(method, params)\n",
    "                waveforms = bench.waveforms\n",
    "                if title in benchmarks_radius:\n",
    "                    benchmarks_radius[title] += [bench]\n",
    "                else:\n",
    "                    benchmarks_radius[title] = [bench]\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "recordings = {}\n",
    "\n",
    "xaxis = [100, 300, 500, 700, 900]\n",
    "\n",
    "for cut_off in xaxis :\n",
    "    recordings[cut_off] = si.bandpass_filter(recording, freq_min=cut_off)\n",
    "    recordings[cut_off] = si.common_reference(recordings[cut_off])\n",
    "    recordings[cut_off] = si.zscore(recordings[cut_off])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "extract waveforms shared_memory multi buffer with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3d7b02c159684cd6938f0995f7149564",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory multi buffer:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "extract waveforms shared_memory multi buffer with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "43d9bc20fcab44e4a4be570570f127bc",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory multi buffer:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "extract waveforms shared_memory multi buffer with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "0109774d5f314b48bd15ecd7e9792fc6",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory multi buffer:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "652e0dab8b2746e382a315367ff1937f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "56eba6e0ec3645a382c42f3005dc355f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5b4622a7af8841b59529ec3a0c3c37d4",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "db482ebb56754faeadf333d3e43106fb",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "2e22d3713d2a4d8ba1f29348acd454ac",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c292ee59fbea4a549e777a91b70dd3e7",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "78c8a450067d4dd9935f24909317f544",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "13947ee20ac04dddae8f8d78dcca86df",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using grid_convolution with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c0930433f0974b238289bf157a77ce31",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using grid_convolution:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "07f9063e880c4a6e985439cac277a65d",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ebda48a42a3a478e866dfb6db0074de3",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using grid_convolution with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "f9bc6c48a64344de818b1cf44c9f1b22",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using grid_convolution:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "extract waveforms shared_memory multi buffer with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "10a7f91472854bf6a3ee57c21cfd74bd",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory multi buffer:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "extract waveforms shared_memory multi buffer with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "72404cfef4e94c60bf5a1349e3e3f71d",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory multi buffer:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "extract waveforms shared_memory multi buffer with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "38669fc2ff864656aca463c1e5e78aac",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory multi buffer:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a066e3d1b71541e493d41ad54225bf9c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "fe745b3205294b538091e3f9b1597480",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5996bfca7a814d7ea409a1e45f80e071",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "96f0f1f2c68342418ad09665a1c13953",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "414ef4eced154dc8af5ef92e72a9e9de",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "689549ef1f434a00836ca9477c56d127",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b8d6ef4ae8f546d7b4ba91a8f9b67c82",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ee93dd341a5947fbbbc5e85932439091",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using grid_convolution with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "7d0e18afaa74492c8c60aa5b0c769293",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using grid_convolution:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "83455afba1314d62ae05fb1c31ec1ed2",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "6141704927ae4d93974ae93392edde7e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using grid_convolution with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4979038b27b44a1381d9c7e117c22d19",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using grid_convolution:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "extract waveforms shared_memory multi buffer with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "c474bdaf1401408f861b1e1e2cbb254d",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory multi buffer:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "extract waveforms shared_memory multi buffer with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "38e6d9e5efba4c7eb6875b3de7853633",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory multi buffer:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "extract waveforms shared_memory multi buffer with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8940c64c11c34a5bbfd0ff61db1d097b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory multi buffer:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a39c52d54dc84cbcbafc218d499adbc6",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b57a10798781430596142c22ab3fa39b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "249c215a638942fc83a8eaa4e196dfc8",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "622196a1d360413fa53efac33db988ec",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "f50218a6a148448c9a408d00f3e98663",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "25112f0960af41f18d7fd9a6d609cbd4",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9cef84c096dd4c92ae05b15186e367af",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/media/cure/Secondary/pierre/softwares/spikeinterface/src/spikeinterface/sortingcomponents/tools.py:80: RuntimeWarning: invalid value encountered in divide\n",
      "  prototype = np.nanmedian(waveforms[:, :, 0] / (np.abs(waveforms[:, nbefore, 0][:, np.newaxis])), axis=0)\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "96860fca5c8245229e14eac9cfe78d27",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using grid_convolution with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "274d88d658d343538b7dbf4a386d0b95",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using grid_convolution:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b922219fa2fb4771b6c5c4ce4a4bd596",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "114c64ae518648a18e00d94af84dad56",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using grid_convolution with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "7252f60dda164522be2276d39d18c71b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using grid_convolution:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "extract waveforms shared_memory multi buffer with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "0931a310cfc7415cb23a4aab5a35a147",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory multi buffer:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "extract waveforms shared_memory multi buffer with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5c346cd04cab4595a51240e52c91df82",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory multi buffer:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "extract waveforms shared_memory multi buffer with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ffa7531a53b04bf48aec50bf4aacb1d9",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory multi buffer:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1c546a14ab5644d587645464c5f801ba",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "eea401f5168a4a90be9312fbe06c0f73",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "0e25c14209b245d484e5df089c0a5fd7",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "84baa40cd866497f9aa80fdd7f293440",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "95098bc7756f4c75b40a72a1c362c22b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e9e8e54488f54ff0beaf1a6cb2406f9c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1accaa54be94437481e2a82536e75ee5",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9365f9c759b64147823976d1bed003c0",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using grid_convolution with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "7ddb1882134e49918d1c0b87ea6b0ce0",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using grid_convolution:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "dda45786556340d785075bad2b055c15",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a5e5a2480768461da7427c9c7dcdf8fb",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using grid_convolution with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "204aa7697b3944f3b27c3ce91e391a74",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using grid_convolution:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "extract waveforms shared_memory multi buffer with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "0fcdb7c5d3cf4ae6bb7a85afe2156abb",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory multi buffer:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "extract waveforms shared_memory multi buffer with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "99dcf26c4d2e4f20bf2c884151fb723b",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory multi buffer:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "extract waveforms shared_memory multi buffer with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "abc2ebc22c3a4029b511f9bfd63a2e20",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory multi buffer:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "7c978ad43c2843f8ae2696901e6cda02",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "31e52105fdbf4ec2bc9644925e3ec9e8",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using monopolar_triangulation with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1c61df0223ee43c2a543869d4e01b4a1",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using monopolar_triangulation:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "746b1ef1184e46efb62c2930ce9a9787",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "09ededa8b6694f6b93ad3a627b6071c8",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using center_of_mass with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e8b8b3d46a224ff581c4da0a7d0804ba",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using center_of_mass:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "209d272781cd40a58896d5e1bbfae3a4",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "874355382a6a47a5b09fef79b4310de7",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using grid_convolution with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "cb7008405dce4061b4129354886ee37a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using grid_convolution:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "cc9ba8e40dab4e709248fdde0541b76a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ef1043ab871d42d8955d6f80ed56ded6",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "extract waveforms shared_memory mono buffer:   0%|          | 0/10 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "localize peaks using grid_convolution with n_jobs = 72 and chunk_size = 9765\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ce4637a702f1484eb20d5097137d49b9",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "localize peaks using grid_convolution:   0%|          | 0/33 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "benchmarks_cutoff = {}\n",
    "\n",
    "benchmarks_cutoff['xaxis'] = xaxis \n",
    "\n",
    "for cut_off in xaxis:\n",
    "    waveforms = None\n",
    "    for method in ['monopolar_triangulation', 'center_of_mass', 'grid_convolution']:\n",
    "    \n",
    "        if method == 'monopolar_triangulation':\n",
    "            for feature in ['ptp', 'energy', 'peak_voltage']:\n",
    "                title = f'Monopolar ({feature})'\n",
    "                params = {'enforce_decrease': True, 'feature' : feature}\n",
    "                bench = BenchmarkPeakLocalization(recordings[cut_off], gt_sorting, positions, job_kwargs=job_kwargs, title=title)\n",
    "                if waveforms is not None:\n",
    "                    bench.waveforms = waveforms    \n",
    "                bench.run(method, params)\n",
    "                waveforms = bench.waveforms\n",
    "                if title in benchmarks_cutoff:\n",
    "                    benchmarks_cutoff[title] += [bench]\n",
    "                else:\n",
    "                    benchmarks_cutoff[title] = [bench]\n",
    "        elif method == 'center_of_mass':\n",
    "            for feature in ['ptp', 'energy', 'peak_voltage']:\n",
    "                title = f'CoM ({feature})'\n",
    "                params = {'feature' : feature}\n",
    "                bench = BenchmarkPeakLocalization(recordings[cut_off], gt_sorting, positions, job_kwargs=job_kwargs, title=title)\n",
    "                if waveforms is not None:\n",
    "                    bench.waveforms = waveforms    \n",
    "                bench.run(method, params)\n",
    "                waveforms = bench.waveforms\n",
    "                if title in benchmarks_cutoff:\n",
    "                    benchmarks_cutoff[title] += [bench]\n",
    "                else:\n",
    "                    benchmarks_cutoff[title] = [bench]\n",
    "        elif method == 'grid_convolution':\n",
    "            for feature in ['gaussian_2d', 'exponential_3d']:\n",
    "                title = f'Grid ({feature})'\n",
    "                params = {}\n",
    "                bench = BenchmarkPeakLocalization(recordings[cut_off], gt_sorting, positions, job_kwargs=job_kwargs, title=title)\n",
    "                if waveforms is not None:\n",
    "                    bench.waveforms = waveforms    \n",
    "                bench.run(method, params)\n",
    "                waveforms = bench.waveforms\n",
    "                if title in benchmarks_cutoff:\n",
    "                    benchmarks_cutoff[title] += [bench]\n",
    "                else:\n",
    "                    benchmarks_cutoff[title] = [bench]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "from spikeinterface.sortingcomponents.benchmark.benchmark_tools import BenchmarkBase, _simpleaxis \n",
    "import pylab as plt\n",
    "import matplotlib\n",
    "\n",
    "def plot_comparison_precision_2(benchmarks, colors=None):\n",
    "\n",
    "    import pylab as plt\n",
    "    fig, axes = plt.subplots(nrows=3, ncols=len(benchmarks) + 2, figsize=(15, 7), squeeze=False)\n",
    "    \n",
    "    to_explore = list(benchmarks_ms.keys())\n",
    "    to_explore.remove('xaxis')\n",
    "    \n",
    "    for title in to_explore:\n",
    "        \n",
    "        if title.find('Monopolar') > -1:\n",
    "            jcount = 1\n",
    "        elif title.find('CoM') > -1:\n",
    "            jcount = 0\n",
    "        elif title.find('Grid') > -1:\n",
    "            jcount = 2\n",
    "    \n",
    "        for icount, benchmark in enumerate(benchmarks):\n",
    "\n",
    "            bench = benchmark[title]\n",
    "            \n",
    "            #vrange = np.array(list(bench.keys()))\n",
    "            #v_min = np.min(vrange)\n",
    "            #v_max = np.max(vrange)\n",
    "\n",
    "            #my_cmap = plt.get_cmap(cmaps[jcount])\n",
    "            #cNorm  = matplotlib.colors.Normalize(vmin=v_min, vmax=v_max)\n",
    "            #scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=my_cmap)\n",
    "\n",
    "            if icount == len(benchmarks) - 1:\n",
    "                b = bench[1]\n",
    "\n",
    "                spikes = b.spike_positions[0]\n",
    "                units = b.waveforms.sorting.unit_ids\n",
    "                all_x = np.concatenate([spikes[unit_id]['x'] for unit_id in units])\n",
    "                all_y = np.concatenate([spikes[unit_id]['y'] for unit_id in units])\n",
    "                try:\n",
    "                    all_z = np.concatenate([spikes[unit_id]['z'] for unit_id in units])\n",
    "                except Exception:\n",
    "                    all_z = np.nan * np.zeros(len(all_x))\n",
    "\n",
    "                gt_positions = b.gt_positions\n",
    "                real_x = np.concatenate([gt_positions[c, 0]*np.ones(len(spikes[i]['x'])) for c, i in enumerate(units)])\n",
    "                real_y = np.concatenate([gt_positions[c, 1]*np.ones(len(spikes[i]['y'])) for c, i in enumerate(units)])\n",
    "                try:\n",
    "                    real_z = np.concatenate([gt_positions[c, 2]*np.ones(len(spikes[i]['z'])) for c, i in enumerate(units)])\n",
    "                except Exception:\n",
    "                    real_z = np.nan * np.zeros(len(real_x))\n",
    "\n",
    "                dx = np.corrcoef(np.nan_to_num(all_x), real_x)[0, 1]\n",
    "                dy = np.corrcoef(np.nan_to_num(all_y), real_y)[0, 1]\n",
    "                dz = np.corrcoef(np.nan_to_num(all_z), real_z)[0, 1]\n",
    "                ax = axes[jcount, icount+1]\n",
    "                print(np.min(real_x), np.max(real_x), np.mean(real_x), np.std(real_x))\n",
    "                print(np.min(real_y), np.max(real_y), np.mean(real_y), np.std(real_y))\n",
    "                print(np.min(real_z), np.max(real_z), np.mean(real_z), np.std(real_z))\n",
    "                #x_means = np.array([np.nanmean(dx), np.nanmean(dy), np.nanmean(dz)])\n",
    "                #y_means = np.array([np.nanstd(dx), np.nanstd(dy), np.nanstd(dz)])\n",
    "                x_means = np.array([dx, dy, dz])\n",
    "                \n",
    "                ax.plot(np.arange(3), x_means, c=colors[title], lw=2)\n",
    "                #ax.fill_between(np.arange(len(x_means)), x_means-y_means,x_means+y_means,\n",
    "                #            color=colors[title], alpha=0.05)\n",
    "                _simpleaxis(ax)\n",
    "                \n",
    "                ax.set_ylabel('corrcoef')\n",
    "                #if jcount == 0:\n",
    "                ax.set_xticks(np.arange(3), ['x', 'y', 'z'])\n",
    "                ax.set_ylim(0, 1)\n",
    "\n",
    "                ddx = np.abs(1 - all_x / real_x)\n",
    "                ddy = np.abs(1 - all_y / real_y)\n",
    "                ddz = np.abs(1 - all_z / real_z)\n",
    "                if title == 'Grid (exponential_3d)':\n",
    "                    ddz = np.abs(1 - (all_z / 5) / real_z)\n",
    "                ax = axes[jcount, icount+2]\n",
    "                \n",
    "                #x_means = np.array([np.nanmean(dx), np.nanmean(dy), np.nanmean(dz)])\n",
    "                #y_means = np.array([np.nanstd(dx), np.nanstd(dy), np.nanstd(dz)])\n",
    "                x_means = np.array([np.mean(ddx), np.mean(ddy), np.mean(ddz)])\n",
    "                y_means = np.array([np.std(ddx), np.std(ddy), np.std(ddz)])\n",
    "\n",
    "                print(x_means, y_means)\n",
    "                ax.plot(np.arange(3), 100*x_means, c=colors[title], lw=2)\n",
    "                #ax.fill_between(np.arange(len(x_means)), x_means-y_means,x_means+y_means,\n",
    "                #            color=colors[title], alpha=0.05)\n",
    "                _simpleaxis(ax)\n",
    "                \n",
    "                ax.set_ylabel('error (in %)')\n",
    "                #if jcount == 0:\n",
    "                ax.set_xticks(np.arange(3), ['x', 'y', 'z'])\n",
    "                ax.set_ylim(0, 350)\n",
    "            \n",
    "               # ax.set_ylim(0, 45)\n",
    "            \n",
    "            ax = axes[jcount, icount]\n",
    "            \n",
    "            _simpleaxis(ax)\n",
    "\n",
    "            x_means = []\n",
    "            y_means = []\n",
    "            y_stds = []\n",
    "            labels = []\n",
    "            \n",
    "            for b in bench:\n",
    "                x_means += [np.nanmean(b.medians_over_templates)]\n",
    "                #x_stds += [np.std(b.medians_over_templates)]\n",
    "                y_means += [np.nanmean(b.mads_over_templates)]\n",
    "                #y_stds += [np.std(b.mads_over_templates)]\n",
    "                #colors += [scalarMap.to_rgba(key)]\n",
    "                #label = b.title.replace('Mononopolar', '')\n",
    "                #label = label.replace('CoM (ptp)', '')\n",
    "                #label = label.replace('Grid', '')\n",
    "                #label = label.replace('[', '')\n",
    "                #label = label.replace(']', '')\n",
    "                #labels += [label]\n",
    "                #title = b.title\n",
    "            xaxis = benchmark['xaxis']\n",
    "                #ax.scatter(x_means, y_means, c=colors, label=label, s=200, edgecolor='k')\n",
    "            \n",
    "            x_means = np.array(x_means)\n",
    "            y_means = np.array(y_means)\n",
    "            ax.plot(xaxis, x_means, color=colors[title], lw=2, label=title)\n",
    "            ax.fill_between(xaxis, x_means-y_means,x_means+y_means,\n",
    "                            color=colors[title], alpha=0.05)\n",
    "                \n",
    "            #ax.errorbar(x_means, y_means, xerr=x_stds, yerr=y_stds, fmt='.', c='0.5', alpha=0.5)\n",
    "                \n",
    "    \n",
    "            #ax.legend(loc='lower right')\n",
    "            \n",
    "            if icount == 0:\n",
    "                ax.set_ylabel('error medians (um)')\n",
    "            else:\n",
    "                pass\n",
    "                #ax.set_yticks([])\n",
    "            \n",
    "            if jcount == 2:\n",
    "                if icount == 0:\n",
    "                    ax.set_xlabel('Time (ms)')\n",
    "                elif icount == 1:\n",
    "                    ax.set_xlabel('Radius (um)')\n",
    "                #elif icount == 2:\n",
    "                #    ax.set_xlabel('Cutoff (Hz)')\n",
    "            #else:\n",
    "            #    pass\n",
    "                #ax.set_xticks([])\n",
    "                #ax.set_xlim(7, 9)\n",
    "            #    ax.set_xticks([])\n",
    "            #else:\n",
    "            #    ax.set_xticks(np.arange(len(labels)), labels, rotation=45)\n",
    "                #ax.set_xlim(12, 14)\n",
    "            \n",
    "            #ymin, ymax = ax.get_ylim()\n",
    "\n",
    "            if icount < 3:\n",
    "                ax.set_ylim(5, 45)\n",
    "            elif icount == 3:\n",
    "                ax.set_ylim(0, 1)\n",
    "            #else:\n",
    "            #    ax.set_ylim(0, 2)\n",
    "                #ax.set_xlim(5, 20)\n",
    "            \n",
    "            #ax.set_title(method)\n",
    "        axes[jcount, 0].legend()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "-253.39537150829057 254.73664441529127 -4.32136246547113 148.86549748421214\n",
      "-251.55885562026796 254.2166298918341 -3.1106822830030185 141.68017737393828\n",
      "0.11247552892432466 29.901469637037284 14.610098530459542 8.747256821809565\n",
      "[0.26176414 0.14755583 1.55152467] [5.86391943 0.52112072 8.36409764]\n",
      "-253.39537150829057 254.73664441529127 -4.32136246547113 148.86549748421214\n",
      "-251.55885562026796 254.2166298918341 -3.1106822830030185 141.68017737393828\n",
      "0.11247552892432466 29.901469637037284 14.610098530459542 8.747256821809565\n",
      "[0.27101734 0.1496237  1.30115639] [6.4672023  0.52332868 6.41413417]\n",
      "-253.39537150829057 254.73664441529127 -4.32136246547113 148.86549748421214\n",
      "-251.55885562026796 254.2166298918341 -3.1106822830030185 141.68017737393828\n",
      "0.11247552892432466 29.901469637037284 14.610098530459542 8.747256821809565\n",
      "[0.21984407 0.15634689 0.93558166] [3.8948855  0.56499817 2.41805591]\n",
      "-253.39537150829057 254.73664441529127 -4.32136246547113 148.86549748421214\n",
      "-251.55885562026796 254.2166298918341 -3.1106822830030185 141.68017737393828\n",
      "nan nan nan nan\n",
      "[0.78525826 0.17811687        nan] [10.84672222  0.58955217         nan]\n",
      "-253.39537150829057 254.73664441529127 -4.32136246547113 148.86549748421214\n",
      "-251.55885562026796 254.2166298918341 -3.1106822830030185 141.68017737393828\n",
      "nan nan nan nan\n",
      "[0.76828198 0.17523924        nan] [10.70668206  0.5869806          nan]\n",
      "-253.39537150829057 254.73664441529127 -4.32136246547113 148.86549748421214\n",
      "-251.55885562026796 254.2166298918341 -3.1106822830030185 141.68017737393828\n",
      "nan nan nan nan\n",
      "[0.77687071 0.28180521        nan] [14.02086439  8.21598848         nan]\n",
      "-253.39537150829057 254.73664441529127 -4.32136246547113 148.86549748421214\n",
      "-251.55885562026796 254.2166298918341 -3.1106822830030185 141.68017737393828\n",
      "0.11247552892432466 29.901469637037284 14.610098530459542 8.747256821809565\n",
      "[0.32634488 0.17056626 3.22760573] [ 7.25461619  0.61954283 13.10051076]\n",
      "-253.39537150829057 254.73664441529127 -4.32136246547113 148.86549748421214\n",
      "-251.55885562026796 254.2166298918341 -3.1106822830030185 141.68017737393828\n",
      "0.11247552892432466 29.901469637037284 14.610098530459542 8.747256821809565\n",
      "[0.26231231 0.15369812 2.70107184] [ 5.08572074  0.55672897 11.54996928]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/cure/anaconda3/envs/circus/lib/python3.10/site-packages/numpy/lib/function_base.py:2897: RuntimeWarning: invalid value encountered in divide\n",
      "  c /= stddev[:, None]\n",
      "/home/cure/anaconda3/envs/circus/lib/python3.10/site-packages/numpy/lib/function_base.py:2897: RuntimeWarning: invalid value encountered in divide\n",
      "  c /= stddev[:, None]\n",
      "/home/cure/anaconda3/envs/circus/lib/python3.10/site-packages/numpy/lib/function_base.py:2897: RuntimeWarning: invalid value encountered in divide\n",
      "  c /= stddev[:, None]\n"
     ]
    },
    {
     "data": {
      "image/png": "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t4pgxnSDtD5pqerXVfUef/JK4Ad4Y0bndFG7+rULvvMdTj31VH5+/fUMHz4cgKFDh/LjK67glFO8vH/V1dVcdOGFjB41ikEDB/KlL32J1asbP4w65JBDCAaDvPTSS4D3If7uu+9uN+gwf/58ysvLOfTQQ5Pz7r33XiaMH88N11/P8GHDGDJ4MN+/5BKi0WhyndWrV3PyySczbOhQhg8bxvlz51JR0RBo/r/LL2ePceMoyM9n/B57cPNNN7XahuXLl7Pn5MnMu/LKNtv61a9+lYsvuqjRvHvvvZfxe+yRHIf96KOPst+++1JcVMR+++7L3/7W8lDnRx55hOuuu46XXnqJgvx8CvLzWb58OWvXruW4445j8KBBFBcVccThh/P2228nt1NVrvvZzxg1ciQDSkv53sUXc8zRR3PVvHnJdT744ANmzZrF4EGDKBs9mst/9KNG566goIADDjiAp558ss33a0xnicZdYu6uBSbW7azl/Q0VCDBjTAnZoczkmXD9nAthx0ECASrq4myvjlLvKmEnQCiNYMLKLVXMe2oJ1/9rORoJM6g4BxFh36F5fP+I0Rw/eeBuvb+mvSMAskIOeU16R2jT3hGO9JbeEa1S1UfxekpUA9PxKu8co6p/zWjDdsHmyjoWrtvB++t3smjDThZvrODDTRUs3VzJJ1sqWba1ihXbqllVXs2a7TV8uqOGdTtr2VBRy6bKOrZU1bGtup7tNVF21EapqItRVR+jOpGkNe56wUK/LKwx/ZGIfD/l+8tbe3Vgl78HJgMnqGpNyvy/AXuJyEkikkXzql/3Az8WkWK/WuF5eIUBjDFdLO1Sognq/av5UBe0pdvdN7/Dw9Z2y9nTR7a7ztKlS/nkk0+45dZb21zvexdfzMKFC/nv669TVFTERRdeyIlf/CLz334bx2n4IH3uuedy5513MnPmTJ5//nkKCwuZfsABbe57wYIFTJgwodF+AFatWsXq1av5ZNky1q1bx+ePP57S0lJ+dPnl1NbWcvRRR3Hqqady//33U1tbyxlnnMFFF17IXXffDcDkPffklVdfZejQobz44ouc8IUvMGnyZI499thGx3n99df56sknc8211zJnzpw22zpnzhy+fs453HDjjYT83BH33XsvZ599NiLCf//7X8484wwefewxjjnmGP7xj39w8le+wosvvcRBBx3UaF9f+9rX+HDJEl577TX++fzzyfmrV6/mW9/6FkcffTQiwo8uu4yvnHQSSz/+mFAoxAMPPMBvfvMb/v7ss+y1117c9Otfc8sttzBjxgwANm3axJEzZ3LtT3/Kk08+yebNm/nS7NlkZ2dzxU9+kjzOXnvtxYJ33mnz/RrTGeKuV9VhVwIT1fVxXlu5DYB9hhYwJL/780yon1vCEcEJCDXRONV1MT/fRCCt97W2vIb7Xl/Nsm01DC3NZWhpGIBJA3OYNXEAJTm7novGTSSzxKu73VKpT9cPWqAggYbKGruSlLQnU9X/Av/NdDt216bKOhZ82nSIetcRgQBCwP99SFRcSUxLcn5L82hlfsvrSJP9tLQ/QQgEUo5J4/YEWjlG03V6W4DNdLvPAr/wv29tjK0CP2tvRyIyGjgfqAM2pPzuna+qD7VT9etKvMDGKryg6vVWRtSY7tFqcEJEvq+qv/C/bzVKqartXiBM+hIJdRI9Jlriui73338/Tzz5ZHK9X990EwNKS3nrrbc45JBDkuueedZZXH311Wzbto077rgjraEa28vLKShoPn48EAhww403kp2dzbhx4/j+pZfyixtv5EeXX87TTz+NqnLV1V5v3ezsbK6++mpmfOYz3H7HHTiOwxlnnJHc12c/+1mO//zn+fe//tUoOPHoX//K9ddfz/0PPMDRRzcdqtzcscceSzAY5Omnn+ZLX/oSy5Yt47XXXuP+Bx4AvEDFl086ieOOOw6Az3/+88z+0pe45+67mwUnWjNq1ChGjRqVnL7m2mv57W9/y8cff8yee+7Jgw88wNy5c9lvv/0A+P6ll3JrSnDpgfvvZ+rUqZx//vmA97P94WWX8aPLLmsUnCgoKGDZ8uVptcmYXeW6Sn08vktP4l1VXl2xlbqYy9D8CHsNaZYfrMslymk64iW8rKqPEot7STAjwfZ7OKzfUcsDb6xm0cYqhg3IY8Qg7z2UFWdx/KQBDMmP7HLb4q7iuooEIOKX+nT8G7JEZY1YamUNB5w0gym9kYhcAzzrBygS8z4DfE5V2+4W561bgtcl+3PAFuBHqvrHFtabh1dVrC5l9j6qutxfvq+/n8nAEuAbqvpuR97LgNww+wwtSPbWSXz1fq4tzWv4XXXbWafpuorXkyaOV3nGuxfrGwRaCFp0PPhSmhNm6rDCdo9nehdVPT7l+yN3c1+r8H7lWlveatUvVa0Dvu6/jDHdqK2eE50WvcyEdz/dwaKNFZB8dgUTnTjl1Q1d6b+455BubVPqsVsTzisC4KPlqxgyeo8W19m0cSN1dXWUDBnRsM9AhAEDB/HhJyuZNHU69TGXOAEC2QV89phjufa663nhhRe48eZbWbJ4UZvtCecWsG37jkbLq+vjDBg4iDpC1PnzBwwdwdq1aymvjrJ46SesXr2a4uLiRvsSET5asYahw4Zz+62/44F772bduk9RVWprajjpq6dQXh1lR423z5///Ho+e8znmHboEWmdL4CTTz2dO+66m5nHfoHb7ribw2ceSV7pEMqro6xYtZqp++7faF/DR5bx3sJ3KK+OUlEba3QuaqIu0bjbaP2tW7ZwxY9+wGuvvsKOHdsJ+MlEl69Zz9Cy8axes5YTvjyi0TZDh4+gJurt58OPvYBJ6rlRVdx449/Hzdu2k1dQ1Or7ro7G+fPCT5PTB44sZvzAvBbXNaYlrl+Zw5GOBybirvLWmnI2V9WTE3L4TFlJtz4FTfSWAK9aRU3UJRZTHEf8PA1tt2VzRR0PvrGGtz/dyfCBeYwe4gVgh+aHOX7SAEYXZ+9y22Kud/MZDAg5WUGC/o2W65c1xdU+3TuiDd8Arm8y7z283FXtBieAW/CS2A0G9gWeEZGFqtrS2O9HVPWMpjNFJAw8AdwE3Ir3JPUJERmvqvVpvg8G52cxuJt6CWmjoAa4+F/90rLedMtBjpaCHm7T/bUSNGm6TvNAS8oyUtqTmG7hWM3aB97fRHLoyq4FXuJu3wnYGGOMadBqcKIzo5eZkBjj6/H+EdMAaA9/AjFu/HjGjBvHo395hMOP/GyL65QOHEAkEmH16pWMGTcOgMrKSrZs3sSwEcNR/z/8r2ed83W+/IXj+Oppp1NQVJg8B62di72nTmXZJx8Ti8eSQzsUZcvmTVRVV5GTkwPAqlUrGTrcO96IkaMYt8d4/jO/5WEJb7z+Glf/5P949OlnmXbAgTiOwzmnn4qrbkp74aG/PsZ3vzmX7194ATfc9Ju0bn5OOf1Mjjh4Ous3rOORhx/kJ1f/NLm/YcNHsHr1ykbvdeXK5QwbPiJ5jlLPRcB/gpm6/jXzfszGDet57qVXGDJkKJUVFYwZOhD12z502DDWrF7VcF5V+XTtmuT5HzFqFIcf+VkefvTxZm1PPc6SxYv43HHHt/pzUYX6eMMy18Ylmw5Q9cpUyi50ra6uj/PKiq1sqaonIF6eiaxuzDPhqpccMu66VNd74/KDjpAVbr8N26rq+eOba/jvynKGD8xnrP+0tTQ7yHGTBjB+QM4uBVkSuSRUvUog4aCXeDP1pqw/9I5oRzZevolU1TSU6muVn6DuJGAvVa0E/iMiT+KVML+sA22YifdZ5yZ/WOpv/HHtn6V5mdMeQRK9BFp/6Nsrqd8rpHnQoq1AS8s9TLJCu5+c1vQ8InIfcKWqrmxjnTLgKlU9u7vaZYzpPh3OOdFbTB1WwOTBeY3GVa9aVklxdog2enn1CLfeciuzT/wiI4cN5TvfuYBhw4axceNG7rnnbsaUjeFrp5zCmWeeyY0/vYYD951KUVERl1/xIyZNmsRRh30Gx3GIOA5BJ0BxdpgvHHsMz/3jn+y5554UZ4fJj3hjqYuzwy0e/7MzDqW4qIgP3307mTchNxTEdV1uuOoKfn79Daxfv57bfnszc84+m+LsMF/78myuv2Yef7jpF1xwwf8jLy+PdevW8b+33mL2l76E1tXgOA5jRwyjJCfCs88+y7+f/wdf+crJFGeH2ZHltWX86JG8/PLLHH/cLC46/1zuvucegsG2f00P3Hdvpk2fzqUXfJuqykpO/9rJRCLe/s79+jkc+7ljOOfsFzn66KP55z//yTNPPsG//v0ixdlh8pqci7IRw3ns07XkOhAOe/PqqqooyMtjzNDBxGL1XH/VFQDkRUIUZ4eZc9ZZ/OhHl3HqV09mypQp/Pa3v2XD+vVkhRyKs8PM/fo5/OG3N/P4ww9y6mmnEQ6HWblyJUuXLmXWrFkAVFRU8M7b87nzjjta/blsDDucPHkY4D01yslQEkLT+6jfY2JXnthvrKjj1RVbqY255IYdDh9TSmluy7+jnS3R7vqYS200DghBJ0A42P6NyY6aKH96aw0vfbKNoQPy2GOE13MpP+Iwa0IpU4bk7VLvhUQ+CYBQMEDE8QIPiRwSARFCDgSd9EqX9nGfAMfSuLLX0cCyNLadAMRUdWnKvIXAEa2sf4KIbMPLsv87Vf29P38KXqK71Gjue/78ZsEJEZkLzAUaDeczu09EksM6evrnMJMxrwNvish7wPPAYmAnXinPPfF6cu8DXJGxFhpjulSrn/BEZIWILG/v1Z2N7YiQEyAr6BBJeQkNTyR68utznzuGV159lSVLlrDv1H0oKizgiMMPY/OmTcw8ciYi8Ktf/5rp06Zx8EEHMqZsNOs3bODxJ54gGHQQIfnvvojXG+Doo49i2LChyWMklrX0CgYdvvOd73D33Xc1rC8wevRoRowYwR7jxnLoIQdz7LHH8oMf/AARyM3N4YV//YslS5YwZc/JlBQX8bljjmbhwncRgVmzjuXMM8/kkIMPYvCggTz22KPM/tKXIOW4iTaVlpbw/AsvsHbtGr761ZOpr69r95zNmTOH5557llNPPZWsrEhy/owZn+Gee+/lhz+4lAGlJfzosh9y/wMPcMghB7d4Lk7+6smMHDmS4cOGUlpSzMqVK7jq6qvYvHkTgwYOYL99p3LIoYfiOE5ym7POPotvf/vbnPCFzzNs6BA+XbuWgw8+mKyI146hQ4fwr3//myeefIJxY8cwoLSEr5z0ZVauWJ7cx5/+9DAzjzySCRPGt/4+gUgwkHz10yexpoMSN/iqHQtMqCpLNlXwwsebqY25DMmPcNykQd0WmKiLxtlZE6W8pp66epeQ45CVRlnQitoYd7+2irkPvsvirbWMH1lMXnaI7GCAEyYP4HuHjWbvofkdDhzEXSUac3FVyQo55GcFkwHCWNwbPpgdcsgOBXpTqc+udh3wiIhcLyLfEJGf4w3pSGc4aB7eTUmqHUBLiU7+jJdPYiBeVv2fiMipKfvZkeZ+UNXbVXW6qk4fOHBgGs00xnQWVf0DMAb4C16vpzvxghR3AkcCfwXGqOrtmWqjMaZrSWslq0Tk9JTJscC38RJKrcC7cHwduFVVr+3qRrZk+vTpOn/+/DbXibtePfnETdzHH33IpMmTu6N5vV5NTQ37Tp3Kk089xcSJE7n33nv52U9/ytKPP85003o813UZPWoU199wA6eddlq769fV1bHP3nvz+BNPMLmN388Plyxh/EQvd1PcL5UYaD9AYXdIvnSuGX1RNBYnrh0rGRqLu7yxupyV5V7ltSmD85k6rKBbbrijsTgV9THqonEcCRAJpXejX1UX42/vrOPJ9zcwsDiX0gIvP0DYEY4YW8xBowoJp1FatKnUfBIRv+wn+OPmgWBACDl9JljY6W9CRI4DLgDKgJXALar69zS22w94TVVzUuZdAsxU1RPa2fYy4ABVPUlELsYrX3p8yvKngJdU9Zdt7ae/XjOM6YA+ceHrDHa9MCYt7V4z2so5kSwXKiKv4NUInp8y7zG8BFMZCU6YrpWdnc1HS5e2v6IB4E9/+hOzZ8/GdV1+ft11VFdXJyuEtCcSidi5Nl0iGneJKe32NkhVURvj5eVb2V4bJRgQDh1dzKjinPY33E3RuEtFXT3V9S6hQICccCitoERNNM6T767n0XfWUVKYzcRRXqJOR+AzZUV8pqyI7A4OgWorn0QiKBF2xIZutEFEgsDNwCWq+mx767dgKRD0E1cmouJTgZaSYTbVkAnbW/8SEZGUoR374CXbNMYYY0wPkm7OiX2Bd5vMe8+fb0yX2nuvvVi1alWz+aNHj+b9Dz7IQIuau/WWW/jWN78JwF577cXTzzzTrHKJMd0pFneJuS7BQPq9BdbuqOG1lduIxpWCSJAjxpZSmB3qwlZ6QYnKunqq6lyCIuSHg8mKOG22tbyGZ97bwAsfbqKoIJsJo0oIBLwx7QeMLOCIscXkRzqWVimdfBKRoBDchTKs/Y2qxkTkFLxeE7uyfZX/EORqETkX7/PGicChTdcVkROBV4DtwAHAd4FECfSXgDjwXRH5A96wD4B/70q7jDHGGNN10v3k9hFwMXBjyryL8J5smH5gzpw5zJkzJyPH7ikBiLa88uqrmW6CMUmuq0Rdr2RoOlSV9zdU8N56b4j/yMIsDikr2aVhEOmqj7lU1kepqo/hECAvEsRpJygRjbv855OtPPPeBj7ZUs2QkhwmjC5NDqnYZ2genx1XQklOxwIqcVdxXUUCkBVyCDlCQMQfGqg4ASE71GeGbnSnJ/Eqbvx1F7f/NnA3sAnYCnxLVReJyGHAs6qaqPpxir9eBFgLXK+q9wGoar2IzMYbs/5zYAkwuyNlRI0xxhjTPdINTnwH+LuIfAdYBYzGSzL1+a5qmDHGmI5zVamLe4GJdJ7u18Vc/rtyG5/urEXwKh1NGZzfZT0D6mJxqupiVPtBidxwqN2eCJ+W1/D39zfwz8WbCIWDDCnNYZ9xA5LLJw7M4eg9ShicH+lQW1LzSeRkBRvlk3DVL1faf8uAdoYQ8KCIfBMv30SivjeqOre9jVV1GzC7hfmvklKOVFVPbbpOk/XfAaal22hjjDHGZEZawQlVfUtExgInAMOBT4GnVbVpBuyeTQRVte64ptdR1xtCHXO9qgtxVwkGIICVEzUNXFXqYy4BIa3rXHl1PS+v2EplXZywE+CwMSUM9RNJdra6WJyK2ih1cRdByA4HCQZaz9kQjbv8d9k2nnlvA4s3VDCkJIfxo0sI+r05soIBpg3PZ/rIQko70FPC8kl0qyjwsP+947+MMaZNfs6aJ4CTVLU20+0xxnSftAfkqupO4KF2V+zBHMchGo0SDndPKTzTT2hL37ZcBUdbXDexrPGc1KlofT0qwsad3r/RcReGFmYRtI/6xqfqlboUSa9k6Ipt1byxqpy4KiXZIQ4fW0peB3M0pNOm+rjLzpoo9X5vjogTJCC02lti3fZa/v7+Bv6xaCPiBBhSmsv+EwYllw8viHDgyAL2GpJHqAPDTlrLJxF3G/JJZIUCOGkGdkzb/JuLJcBvVbUm0+0xxvQefs6aaUAs020xxnSvtD6JiogD/Ag4GxikqoUicixereE/dGUDO1NBYSGbNm1i2LBhaSVcM32ENvrC7gYOWt66leO2co/T3q2PJNYRwXVdtmzeTFZuAdkh70+2ss7+vTYNEkEA0ghMuKos+HQHH26qBGBsSQ4HjiruUEWPdNuzo6aeWFxxAgEiQQfxgxJN2xiLu7y+fBtPv7eB9z7dyeDiHMaOLCbiV9kIBoR9huZx4MhChhV0bOhGm/kkXMURIStoQzc6m39zcbmq3pDpthhjeqUH8BLq3pThdhhjulG6j8muAY4GfoiXdArgY7zkUr0mOFFSUsr6dZ/y8dKlzZ5Sm56hzZ9KOz0UtOm60s50U+3cm0gbU10tnJVNwYAB7a9o+qVY3Bvu094Ndk00zqsrtrKpsp6AwPQRRYwfkNtpPQVUldponIq6KFFXCQUCRIIBXLzSnk6T3hIbdtTy9w828twHG4ghDCnJYdrEQcngRWlOiANGFrDfsPwOlwNtNZ+Eq7gCQUfItqEbXe1FETlCVV/OdEOMMb3O/sCFKfnuUnPWfC5jrTLGdKl0gxOnAYeo6noRudOftwIoS2djEYkAt+IFOEqAZcCPErXPReQovJrjo4A3gTmq2rx25G4KBAIMHzGys3fbpyWCOEpDzwJXXf9rw3qu662rKK6/rvrjul11cRW8pd52mtKbQRVE8POBeDNFBBdFlIbAgkijHgUAif4vIg3hguTNht109Fo95ZrRG0TjceJpBCY2V9bxyoqt1ERdskMBDh9TysC8jvVCaI2rSm00RkVdjLgflMgOOriqIBBO6S0Rd5U3/F4S76zezoCibEYNLSQny8sbIcDkQbkcMLKAsSXZHQqctJdPQkQIBy2fRDdaCTwhIn+leULMn2WoTcaY3uEV/2WM6UfSDU7k4JXyShUG0k1SEwTWAEcAq4HjgT+LyN5AJfAYcC7wFF4vjUeAg9Pct/G1FEhQvA/qXuCA5PeuajJIoHilB735Lol4gJsaQPA/UmoiCOAqBAT1AwiJ+EHAX59AIBlISMYKEAI03EQlbw3sJsE0Z9eMNMTiLjFXCbYxTE1V+XhLFfPXbsdVGJQX5rAxpR3uidCSuKvURGNU1sdw/aBEKOigfjAgKJLsLbFxZy3Pvr+RZz/YSE3c9XpJTBqULB+aH3aYNqKAaSMKKMzqWO6LtvJJJEqBWj6JjNgXeAcY578SFLDghDGmVap6VabbYIzpful+AlwAnINXJzzhNOCtdDZW1SpgXsqsp0VkBV5pr1Jgkar+BUBE5gFbRGSSqn6YZvt6pbaCCQ3rgOu6uKQEE/wyd95ncS+oEHe9J5ReoMG733eTvQ687gjqj7uGRGDA/7/fIwGBAIkeCUJqrjlJ7aJgTBeza0b7vBtvt80eEzFXeWt1Ocu3VQMwaVAe+w8v3O1eAzHXpaY+TlV9DFe9oETYz87q+j2gwgFBlWQuifkryykpyGLooHwKchuSEo8pzuLAkYVMGpTbobwPieueCwSkIZ+E4F37oq43pCPLsXwSmaKqR2a6DcaY3ktERuLdb4zEe2DxkKquzWyrjDFdKd3gxPeBl0TkFCBHRJ4CpgO79MFDRAYDE4BFwLeAhYllqlolIsuAKcBu3Wioeh/eXU3vg2kyQNAo+WFDAKH5ei3sIzUHQmIYhNsw100EJLRhOtm7wR/SgDb0ffV6RWtDTEDEG+oASEBShjI0RBISFRysZ0LPlLipjMaVaCxObcwbp18Xd6mLNbzqY9569XF/fX+7mKvURl2OmlDK4eP7Rx6K7rpm9BauKvXxeLMcDqkq62K8snwr22qiOCIcPLqYMSU5u3XMWFypro9RG4ujqoQCDgHHO76q4iIERdhWVc+zH2zkuQ82srMuzuCSHKZNHEwo6F2nIkFh32EFHDiigIF56VdPSgQkEjHXYDBA2PF6RHht9K6dlk+i5xDvF/RAvJuL1cD/1JI+GWPaISIzgOeA9/CGdu4HXCEix6nqqxltnDGmy6QVnFDVD0RkMnAW3of/VcC5qrqxowcUkRBeSdL7VPVDEckDNjdZbQeQ38K2c4G5AKNGjWr3WNG4UlkbI+h3AUgNHCQ+sqZ+RGrpc2zTWaqKSuLDeKMFycCD6x0sGWxQFMGbDvhDIBqCCeL1VkgMfUjt2WAfrHeZF5hqCALEXDc5HUuZXx+LUxfzKgvUxV3qY/78uNtk+4ZX3L9BirsNPVhcTcm1QQv5NkX8vBhCIOD1VgmIENiNJ7rvfrqzXwQnuvOa0Ru4qtTHvLKcrQUm1u+s5dUV26iPu+RFHI4YU0pxTsdLKMddJRZ3qY3FqYt6uS0QCEkgGZRItMlV5d3V2/n7+xt5c8U2CnMjDC7JZY/8SLKdQ/LDHDiykL2H5BEJplcxqa2AhPhVNxL5JEIOlk+iB/Gfej4FTMYbGjoIWCIiX1TV1RltnDGmp7sB+K6qJhLxIyLnADfSD4dxGtNfpD2wV1U3A7/cnYOJSACvNFA9Xnkg8MaPFzRZtQCoaKENtwO3A0yfPj2tJy+BAISSH6Ibf2BN5GHwvve/+hNu6jI30ZPB/4Cc/JrItZAYJOF9dfxkCwJef+M+LJGAruFGvvHNf+OvLc1rHCjwggIkewhE4y5x9bqnx5OBgRaCAon2AOAFAhI3/wERAoL3vf+kuWFZR8eg+z/cAMlsnCnf7tL5S+QBSbZftSHxJ977kMQh/TajMG7Arj8F7y0ycc3oydQPTEgrv7eqyqKNFSxctxMFhhVk8ZmykrQDAdAQkKiOxvy/ZSUQAEcChAONj6uqbKqs51+LN/Hcoo2UV8cYXJzNfuMHEgl7/7w4AnsNyeOAkYWMLIyk9feWDEioF7BtLSCBX40j6Fg+iR7qZuB/wGf8Hk55eJ8jfgPMzmTDjDE93mTg3ibz7gd+1f1NMcZ0l7SDEyJyCN5QjkZPJ9PNuO137bwLGAwcr6pRf9Ei4OyU9XLxEmctSrdtrUkkSauPxRsFGhqebLdeTzJlFAUIOJCSeKHniPlDAaLxxFAAbxhAfSxlOuZS7/cKaDovsW1bAYS4C3HXCxIkeg0kuk+r0hAACHi9AlKDAk6jZY2DAs3npd5YBPy7cScxRQAIdcE5TAQGGgcEvLwbIt7NVSAgOP77CQbAESHkeNMhJ0AoIIQd7/tw0JuOBL3vI8EAWcEAWaEAkaBDJBjwbqh2KTjiqayLMbQwq/NOQg+UiWtGT6bq9ehJBN6aqo+7vL5yG2t2eHmK9xlawN5D8tP6/Yq5LrG4S000TszvKRQQr4dXJNi8h0bcVd5eVc6zH2zkrRXl5GaFGFKayx6jspLrFmcHmT6igP2HF5Abbj/5ZlsBCfCGbFhAoteZAYxW1RoAVa0UkYvxKncYY0xbNuKVE52fMm9/mifoN8b0IWkFJ0TkWry8EwuB6pRFHcm4/Xu8KOjRiQ8qvr8BN4rIScAzwE+A9zojsZ3rKjXROFl+EoaGqhGJD7SNnwB6N93+y38y5/pP65NP6v11Gp7eK67bEAhJrBdPrpfyhN8fs908mJASMEgMMUgNIjRZJ3V71y/D6QQEJ+AlfgsExJ8WHKeFeanrOQ3BAScgBJwggZAQESE7A70+BD8w4N+AeQEBLxgQDCSCAd73IT8wEHaEUND7Ggk6RBzxAgROILle0N8mmPJ9yJ/O5M2Nd6/V6x/od5Vuv2b0VInAhLZSMnRHTZSXl29lZ12MkCN8pqyEEYXZbe6zaUAi7vfYCYj3t9T070JVWbe9llc+3sJzH2xiS1U9Awuz2HvcAHKzGsKGEwbkcODIAvYYkNPu8Iq0AxLg/c06gV0O6JmMqAUKgdS/30K8nlDGGNOWm4G/i8htwAqgDDgfuCqTjTLGdK10e06cDxyoqu/tykFEZLS/jzpgQ8oHy/NV9SH/JuN3wIPAm8Apu3KcVPe+toq/vP0psbjbKDgQbyGQkJqwsjs4flCgxWBCk8CBExCcUJBIxAsWNFrPaanHQSe31Q98JG7sEzf6iaBA2AkQcsQrIZgSAPC+TyxrHBBIBg0S8/yeB3bTkT6nj5+mTFwzerKY23pgYnV5Nf9dVU7MVYqyQhwxtpT8FkpxeuU9E72qGgISAUDE+xts+ve3sybKu2t28M7q7SxYvYNNFXVkR4IMKclh7MgiAn4Z0JxQgGkjCpg+vIDinLb7N7UUkAg5AYItBST8a4tdG3qtvwF/E5H/w+stUYZX+vfRDLbJGNMLqOrvRWQ7MAc4Ca9ax0Wq+nAm22WM6VrpBidqgMW7ehBVXUXz3JKpy18AJu3q/ltSURtjvd+9OV0B/0Y8kMxJ4D2xT3TpTwxFSMwLOl7XYu+VCDIEGuc2SHzg9/MGdEWSS0cg7D9xTPQaCDvecIJEACGc7FngLUt8n+x54HeTDqUEDTLdq8C0rq8n/cvENaOn8vKveD0HUrmqLFy3k0UbvVQbZcXZHDyqOJkAGFJzwngBCdf1Ah3epah5QKI+5rJ4fQXvrN7OO6u388mmKiQgFOSEKcoLs9+QArIjDf9sjCrK4oCRBUwZnNesfanSDUiICI6DBST6jsuAm/B6OEXwgo33+fONMaZFIhLE6zlxiQUjjOlf0g1O/Ar4MTCv65rSuc44ZCSzpgyiqj5OdthJBhsahgr4iQUF6mPeUIuamNfFuTra+GtN1KU68bU+To0/tGJXCF4gIdIkaNAomBAMEHGafN/KemG/F4IxpvdKJufVhvLCriaqvyhOk5v02lic/6zYxoaKOgTYf0QhkwbmISKNAhK10TjqQky961VAAgQDgeRNv6qyYnMVC9Zs553VO/jg053Ux1zyckIU5UXYa2wpedmhRkGCsCPsMzSfA0cWMCQ/0uZ7aisgEVeIu15JZAtI9Fn742Xc/yYwEK/Kzli8koD/zWC7jDE9mKrGROQUGhJhG2P6iXSDE38B/i0iF9EkEY2qTujsRnWGmpjLppoYW6vqibpKTbSFwEN9nNqYu0tDOgTIDgXIDjnkNPmaHXbICQbIDjtkhwLkhBq+tjSW2xjTt6VWY9GUAERqgl5J1gDCLzvrf22SiHdrVT2vrNhKVX2crGCAw8aUMigvTMxNlMZNBCS8PTcNSGyprOOd1TtYsHo7767ewfaaKDmRIIV5EcYOL6QwN9KoxK0AwwsjjC3JZmxpDiMLI4SclpMDpwYkEAilBCQUv4eEH5AIOySHsZk+6zbgi+r9AWyCZPDpNmDvDLbLGNPzPYk3nOOvu7KxiFyANyRkb+BhVZ2Tsuwo4BZgFN7Q0Dl+j01EJIKX8+oreHn2blBVqxBiTDdJNzjxCLAWr3tmddur9gxvrNrBvz/Zlta6WcFAQ2AhEUwIO2QHA+S0EGDIDnm9F/pyt/q+qK3Ek60taW2T1tfXVjdqqBLTfCduyvdNf61S++gI4pW6beX4JnNSez94N+LaUCqWhkLEqZWAGlfeaP968smWKt5aU46rUJoT4tCyEoIilFfXJxPxNgQkBJEA1fVx3l+7g3fWeAGJNdtqiIQcCvPCDCzNZUJ+pNFQEICBuaFkMKKsOIvsUOvVNtIJSMTUGyJnAYl+Z5SqLk+doarL/JwyxhjTlhDwoIh8Ey9nTfKjj6rOTWP7dcC1wLFAMkO0iAwAHgPOBZ7Cy4PzCHCwv8o8YDwwGhgCvCgii1X1ud17O8aYdKQbnNgXGKCqHUvikEGjirOYOjQPR4T8rGCjwEKjXg6h3hdkaOkmu+mcpqu0eLvcxk1068duPJ166louzCqtH6uNbROLvHv9hqVNj6ep62vKNq20UaRpgCBRRjTlSXGgYf/JNCHasE6jY6buX7xeMSINe2v6fFkSuUdoXA4ytZJM431L4xVSloX7elbMHqh5AMJNDr8AQBRVafi9ITUAses/r7irzF+7nY+3VAEwpjibCQNyicZc6lL+PoIBwVVh6cZKP2/EDpZsqEAECnMjFOaF2X9CHlnhxpf+/IjD2JJsxpXmMLYkm4IWEmo2PQ9tBiRcJYYkAxJ9PUeKadVmERmlqqsTM/zARHpPDowx/VkUSOSbcPxX2lT1MQARmQ6MSFn0ZWCRqv7FXz4P2CIik/yqX2fj9aQoB8pF5A68HhgWnDCmG6QbnFgCFAPru7AtnWr6iEL2HJjLtqp6wsFAqzfccVeJoy3fO6d74w27dPOdukniWNpkndRdt3Cf2qwtqet4XcNbXNlfxxufnjxOk321+EZo47wkb/RbOl9ejg9tmExOJL6VlCM3rOfnBmnWCj9wkNqelBUSN0Kp2yRu9KWFG/2WtHUv1dJ+Te920yvLCTmC63djCfjla7OCAfIiDvmRIPlZQfLCjpcEN7XyTMArZ+uVpw00+9vbnXKxVfVxXlmxlW3VUQICew7KZ3hBJFmGNgBs2FnHgtVeVY2Fa3ZQE41TkBumMDfCXmNKyckKNvo9jQSFsuJsxhRns8eAbAbmhtv8PfYqfSR6BtFiQEJTe0gExQISBrxqHQ+IyPnAx3hPI2/Fe2ppjDEt8hNiLgF+26SUeGeYAixMTKhqlYgsA6aIyEZgaOpy//vZndwGY0wr0g1O3As8KiK/ADakLlDVHpvUygkIWREnmTCy6U130w/ju/s5ur3NW1re9Jipk9o0KkEbPSJaWDe5gjT0Amgyu+GGvZW2tnTj3/QZcEdu+u2G3/RUWf5wraZcYGd9nJ31caioS3t/iYo3TkAI+kl4E9OOeNV+HElZp8n6QUeojbksXL+T+riSFQyw79ACCrKCVNTEeH/dzmRAYuPOOvKyvSSWZcMLKchpHGxwBEYWZVFWnMWYkmxGFGa1WY3HVcV1/eFDCoEABAOB5JAMJyUgEVcvkBOyHhKmuSuBu/GqfSX+uforcEU6G4tICXAX8DlgC/AjVf1jC+tdive0c7S/3q2qemPK8pXAYCDuz/qvqn5uF96PMaYb+AkxL1fVG7pg93l4yXlT7QDy/WWJ6abLmhGRucBcgFGjRnVuK43pp9INTvzW/9o0KY3SwW5W3SkQELKCTrdVs0h2+U5O+1/9CW0hcJAYQNe894KXCK/RDTveTUJiZuptlEigxUBDa8EPu/k3prEZo4tZsa2aTRX1bK+NUhtVYv7ftBNoXFI4Ue0nIILjVwHyphvKEcdcr2pGZyjODhJ2hSfeWZ8s8ZkVCVKYG6a4KIey4UU4TfJGDM0PM6Y4izI/GBEJBpJtS+VqojKIJoOWTkAIh/xghDT0bkp0nEgEJCKhQPI8GNOUqlYBX/MT05UBK1W16U1BW24B6vECC/sCz4jIQlVd1GQ9Ac4C3gPGAf8UkTWq+qeUdU7wSxAbY3qHF0XkCFV9uZP3WwkUNJlXAFT4yxLTtU2WNaOqtwO3A0yfPr1z/sE3pp9LKzihqi2nZu+D2gswtPboX/E+oCdzDQgQaOhlEAg4zXsnWODAmB5jr2GF7DWsMDntleR0qY8rddEY22uibK6Msamyni1V9ZRXR9lWF6W63qU2rs2GZAUEP6DRcG1wXRfXTSTK9O70RZSAeKU0cyJeMt5IMEDIDyas31bLHQs34Pp5I4ryIkybNIhwsHFcuCQ7yJiSbMaUZDO6KEKWn8QyUTo5MYwr7iquf2z1AypBJ0DIcfBin4r6tUPAC0SIv5+gn9BSLCBhOsAPSHQkKIGI5OJl6t9LVSuB/4jIk8CZwGVN9p/6dPUjEXkC+AyQGpwwxvQuK4EnROSvNE+I+bPd2O8ivJ5WQPJaMw4vD0W5iKwHpgLP+6tM9bcxxnSDdHtO9Freh3H/e29GhwMMichM0wBDahJDCygY07eICCHHIeRAbjhISW4WZSUuMZdkyc76uBKLK67GqYkpVfVxdtTGqaiLs7M2xraaKOXVMXbWxaiPN+oX1Wx8VHVc2bwjSl19LXXROHXROPVRl5ysIJPHlJIdaXy5zg07jCnJYkyxF5AozAo29H4QCOAlX1WFqJ+1MzE8I5ysmKEI0qjUKRLwAyoWiDAZNQGIqerSlHkLgSPa2ki8f4wPwytXmuoh8eryvgNcqqoLm21sjOlJ9sX7ex3nvxIUaDc44eetCOIn0xSRLCCGlwvnRhE5CXgG+Anwnp8ME+B+4MciMh+v19Z5wDmd8YaMMe3r08EJbxy0133BAgzGmN0VCAQIByAcDJBHyE8UqcTjSl0sRl3MSxwZi8eJ++MgnICACLGYy04/aLGjLs722hg7amLsqPVelfVxssLBZpU0EsKOMLo4mzF+3oiBuaHkNctN9Iig4VqnflAhGPSSdKKaDDIkeoQFJEAgAI54uSIsEGF6kDxgZ5N5rY79TjEP78/gnpR5pwML8P6pvxD4h5+Zf3vTjW0MuTE9g6oeuZu7+DFe3puEM4CrVHWeH5j4HfAg8CZwSsp6VwK/B1YBNcD1VkbUmO7Tp4MTTkBwAj02JYYxppcT8ZNXBiASclDVZA6HqKvUR2PEXIjFXQJBocQJUpLjXXbF7z0hfiWbuKvsqIuzoybG9tooO2pj7KyNU5wTZExxNsMKIn5vB8APRsRxUVe8HhGOkOU4BJ2G3DMiDcMzvGSW3teAWEDW9HhtjQtvkZ/b4izgMFVNZq9V1ddSVrtORM7G613xVNN92BhyY3oOEXGAg4CRqvqIiOQAmk4FD1WdhxesbGnZC8CkVpbVAV/3X7ts9dZq8rKClOSGd2c3xvQ7fTo4YYwx3UnES5rpACEHchIBCz85ZjQe94IVqsRdF3UTFS9cEKEgy6Ewy2E0ES9oEUj0dPCCHlFXUdcbuhF0hOyAg+M0TnTptcGGZZhebykQFJHxqvqxP6/Vsd8i8nW8XBSHq+radvbdtICVMaaHEZFxwNN4pT2DwCN4lXu+gtcLokf788J1OI5QWRsjJMLQgiwmDMpln+EF5EZCmW6eMT1Wu8EJf8zWE8BJqlrb3vrGGGMaiAiOIzhOQ+8KxetNEVclGnOJqYMbd/3Snd7wDE2p9hFAksMvHMcv6+mI1wsigF9FxIZlmL5DVatE5DHgahE5F2/8+YnAoU3XFZHT8cagH6mqy5ssGwWMBP6HN9zj/wEDgNea7scY06P8Fi+p7TXAVn/eS8DNmWpQR+RmBckKBSjK8QIRLvDh1ire31RBdV2cgMKAvDBjS3PZc2g++ZGg9WY0hjSCE36t4Wl4SWSMMcbsBvGT7iaqbWQFvWCF63o9I+KuVyXEdUHxq3oIBAMO4WCAUCCQLF1qH2RMH/dt4G5gE97NybdUdZGIHAY8q6p5/nrXAqXA/1L+Jh5U1W/i5aj4PV5CvVrgXeA4Vd2KMaYnOxD4oqq6IuLntNftIlKU2Wal58BRxSzfWs26nbXUxVwiwQDZoQAhJ0Bhjpcdqk6VJVsqWbKlkmjMRVUpjIQYWZLN2NJcSnJCRII2PN30L+kO63gAuAC4qeuaYowx/U8iWBFwvFKd4Jc0Voi5rjeEI+D1kjCmP1HVbcDsFua/ipcwMzE9po19LAL26Yr2GWO61E6gCNiSmCEiw4CNmWpQR0wbUci0EV55cleVLVVRVpXXsHxrFavKa6mNxcnyAxZZIa98OECN67J0SxVLt1R527pKTshhaEEWwwqzKM4OUZQd8hL+G9MHpRuc2B+4UES+g5e9NrXW8Oe6omHGGNNfiT9EI2wJfY0xxvRPjwF3i8i3AUSkFO8h6Z8y2ahdERBhUF6YQXlhDhjZELDYWFHPqvIaPt5cxYpt1dTGlKyQkB3ygxZBL6dUbdxlRXk1K8qrk/sMBYTS3DADcyNewCInRGFWiGDAelSa3i3d4MQr/ssYY4wxxhhjutIVwJ3Aan96E/BHvPwyvV5AhKEFEYYWRDh4dBEAMVdZv7OOVeU1LN1UxbKt1dTE3IZgRShAdlDICgWIurChoo4NFXWN9psfCVKSE0r2sCjODlMQCSYTbBvT06UVnFDVq7q6IcYYY4wxxhjjlws9XUS+C4wBVqnq5gw3q0sFA8LIoixGFmUxY0wxAPVxl093eAGLJRsqWVVeQ2XUJcsPUmSHAmQHvcBFVlCoqItRURdjVXlDtdWAQGFWIljR8NWScJqeKO1SoiIyEjgNL+v1GuChNMp1GWOMMcYYY0yH+clr+20C27ATYExJNmNKspk5rgSAupjLmu21rNhWzaINlXy6o5aamCLQqHdFVtALWGSHHcpropTXRFmRsm8nIBRlNQ5YFGWHyA07FrQwGZNWcEJEZgDPAe8By4D9gCtE5Dg/MZUxxhhjjDHGmC4UCQbYY0AOewzI4ZgJAwCoro+zenstn2ypYsmGStZX1FOfKEcuJJNvhh0h4gi5kQBZIYet1fVsra5vtP+QI36gItwocJEdsjxYpuul23PiBuC7qnp3YoaInAPcCByczg5E5AJgDrA38LCqzklZdhRwCzAKeBOYo6qr0mybMaaPseuFMcYYY0x6csIOkwblMmlQLl/YcxAAFXUxVpfX8uHGSj7cVMXmqnpi2rCNI15Pi7ADQRHyIw75WQ4QYFNlPZsqGwctsoKBJkNDwhRlh4gErXKI6TzpBicmA/c2mXc/8KsOHGsdXi3yY4HsxEwRGYCXkfdc4CngGuAR0gx6GGP6JLteGGOMMcbsovxIkClD8pgyxKu8rKpsr42xalsNizZU8MmWarZWx6jy199cHQcgGICgeMGLwiyH4pwQWWGH2pjbYhLOnJDTbGiIlTs1uyrd4MRGvHKi81Pm7Y+XOTctqvoYgIhMB0akLPoysEhV/+IvnwdsEZFJqvphuvs3xvQddr0wxhhjjOk8It5wjeLhIfYdXgB4AYut1VFWbKvm/U8rWL6thh11cWL+NlWVcdZVekELNx7HESjNDjIwP0JBdhAXpToapzoa59OdtY2Olx8JNkvCWZgVwrHKIaYN6QYnbgb+LiK3ASuAMuB8oDOqeEwBFiYmVLVKRJb58+1mwxiTyq4XxhhjjDGdQEQYkBtmQG6YA0YWAeCqsqmynk+2VPP+up2sKq+lsj5OwHFQYEudsqWuFlWlpi6GgzIoL8ywoggD8iI4jlBZH0tWDlmzvaFyiAAFWcFmOS0KsoIELAmnIf1Sor8Xke14Y8BPwqvWcZGqPtwJbcgDmpYG2gHkN11RROYCcwFGjRrVCYc2xvQyaV8vwK4ZxhhjjDEdERBhSH6EIfmRZEnTuKtsqKjjw41VvL++IlkhJCcrBMCOGOzYUodurqW6NoYbjzO0MMKYkhyGFUWIhJxksGJHrfeiSbnTotRypzleTos8qxzS77QbnBCRIF7PiUs6KRjRVCVQ0GReAVDRdEVVvR24HWD69OnadLkxps9L+3oBds0wxhhjjNldTkAYXpjF8MIsjppQCkA07rJuRy3vr69k8cZKNlTUUReH3OwQEKIyDu9vrmHhxmqqa6MEgEG5ISYOymXMoByyww47a2Nsr4lSWR9nW02UbTXRRscNBqTZ0JCi7BA5IQta9FXtBidUNSYipwAXdFEbFgFnJyZEJBcY5883xphUdr0wxhhjjMmwkBNgdEkOo0ty+MIUr0JIfcxl9fYa3lm7k6Wbq9hcFSUqkJcTBqBS4e2N1by1vpLq2hjhgDAsP8ykwblMHpJPdsRhux+wKK+ppybqsqWqni1VjSuHhB1JVgtJBC2Ks0NkWbnTXi/dnBNP4g3n+OuuHsjvgREEHMARkSwgBvwNuFFETgKeAX4CvGfJ7Yzpv+x6YYwxxhjTu4SDAfYYkMseA3KT82qicZZvrWbB2p0s21Lt9Y4IBMj3AxZbo8prayt5edVOauti5AQDjCjKYp+h+XxmXAGhUIDymijbq6OU10Ypr45SH3fZWFnHxsrGlUOy/XKnRdkhinMaclqErXJIr5FucCIEPCgi3wRWAm5igarOTXMfPwauTJk+A7hKVef5Nxq/Ax4E3gROSXOfxpi+ya4XxhhjjDG9XHbIYcqQfKYMaUgPVlkX46NNVbyzdgcr/AohQSeQ7GGxrjrGumXlPPXRVuqjMQrCDqOKs9l/RAEzJg8i6DiU19T7PSyiya81MZeaijrWNyl3etCoYvYc3GJ6MtPDpBuciAKJfBOO/+oQVZ0HzGtl2QvApI7u0xjTN9n1whhjjDGmb8qLBJk2spBpIwuT83bURnnv0wreW7eT1dtrqYy6hIIBQsEwcWDFznpWLN7Cwws3Eo+7FGc5jC3NYdrIQo6ZMJCwE6CyPp4SsKinvCbKjpoo+ZF0b3lNpqWbEHMJ8FtVrWlvfWOMMcYYY4wxJl2FWSEOG1fCYeNKAFBVtlZHWbBmBx+sr+TTnbXUxpRwyIGQQy2weGsti7fWcudb6xBVSnNCjB+Yw0Gji/hMWSlOQHBVwVKi9xrpJsS8XFVv6I4GGWOMMcYYY4zpv0SEAblhPjdpIJ+bNBDwAhbrd9bxv9XbWbKxig0V9dS7SiTsdeqviCsLNlSxYEMVtfUxQgKD88IcM2kAnxlbmsm3Y9KUbh+XF0XkCFV9uUtbY4wxxhhjjDHGNCEiDCvM4sS9h3Di3t48V5UVW6v53+odLN1UxZbqKDEgK+zd5m6pc3lnzU4LTvQS6QYnVgJPiMhfaZ4Q82ed3yxjjDHGGGOMMaZ1ARHGDchlXEqFkLirfLixkv+t3s6yLdUcOrY4gy00HZFucGJf4B1gnP9KUMCCE8YYY4wxxhhjMs4JCFOG5jNlqFXo6G3SKvqqqke28vpsVzfQGGOMMf2PiJSIyN9EpEpEVonIaa2sJyJyvYhs9V/Xi4ikLN9XRN4WkWr/677d9iaMMb1SutcfY0znSis4ASAijogcKiJf86dzRCS765pmjDHGmH7sFqAeGAycDvxeRKa0sN5cYDYwFdgHOAE4H0BEwsATwINAMXAf3jDVcFc33hjTq6V7/THGdKK0ghMiMg74APg7cJc/+3PAHV3ULmOMMcb0UyKSC5wEXKGqlar6H+BJ4MwWVj8b+KWqrlXVT4FfAnP8ZTPxhrDepKp1qvobQADr+WmMaVEHrz/GmE6Ubs6J3wJ/Aq4BtvrzXgJu7oI2peXtt9/eIiKrMnX83TAA2JLpRvRwdo7al+45ek5VZ3V1Y3oDu2b0aXaO2pfOOepJ14sJQExVl6bMWwgc0cK6U/xlqetNSVn2nqqmVrl/z5//XNMdichcvJ4YAJUi8lGTVex3rWewn0PP8IGq7pXpRnSBtK4/dr3oNezn0HO0e81INzhxIPBFVXVFRAFUdbuIFO1mA3eZqg7M1LF3h4jMV9XpmW5HT2bnqH12jjrOrhl9l52j9vXCc5QH7GwybwfQUnazPH9Z6np5ft6Jpsva2g+qejtwe2uN6oXnsU+yn0PPICLzM92GLpLW9ceuF72D/Rx6jnSuGenmnNgJFDXZ+TBgY8ebZYwxxhjTpkqgoMm8AqAijXULgEq/t0RH9mOMMWDXDWMyJt3gxGPA3SIyAkBESoGb8IZ6GGOMMcZ0pqVAUETGp8ybCixqYd1F/rKW1lsE7JNavQMvaWZL+zHGGOjY9ccY04nSDU5cgRctXI3Xg2ITUAf8rGua1ae12v3LJNk5ap+do/7Dftbts3PUvl51jlS1Cu/ByNUikisinwFOBB5oYfX7ge+JyHC/V+clwL3+speAOPBdEYmIyAX+/H/vYtN61Xnsw+zn0DP0yZ9DB68/bemT56cXsp9Dz9Huz0Ia54hqZ2Wvx8QYYJWqbt6NhhljjDHGtEpESoC7gWPwknFfpqp/FJHDgGdVNc9fT4DrgXP9Te8EfphIgiki+/nz9gSWAN9Q1Xe69c0YY3qV1q4/mW2VMX1fh4ITxhhjjDHGGGOMMZ0t3WEdxhhjjDHGGGOMMV3CghPGGGOMMcYYY4zJKAtOdDIRKRGRv4lIlYisEpHTWllvnohERaQy5TW2u9ubCSJygYjMF5E6Ebm3nXUvFpENIrJTRO4WkUg3NTOj0j1HIjJHROJNfo9mdltDTacRkZdEpDbl5/hRyrLT/OtJlYg87o+F7fPa+jsQkaNE5EMRqRaRF0VkdMqyiH+92OlfP77X7Y3vJq2dIxEpExFtcm24ImV5vzlHxhhjjOkdLDjR+W4B6oHBwOnA70VkSivrPqKqeSmv5d3WysxaB1yLl2ioVSJyLHAZcBQwGhgLXNXlresZ0jpHvteb/B691LVNM13ogpSf40QA//pxG3Am3nWlGrg1g23sTi3+HYjIALxM6lcAJcB84JGUVeYB4/GuG0cCPxCRWd3Q3kxo71pRlPI7dU3K/Hn0n3NkjDHGmF7AghOdSERygZOAK1S1UlX/AzyJd1NhfKr6mKo+jpf9uC1nA3ep6iJVLQeuAeZ0cfN6hA6cI9P3nQ48paqvqGol3g35l0UkP8Pt6nJt/B18GVikqn9R1Vq8G+2pIjLJX342cI2qlqvqEuAO+ui1YzeuFf3mHO0OERknIttEZH9/epiIbLYeat1PRC4VkUebzPuNiNycqTb1RyLytSY9supE5KVMt6unsGtGz2DXi56jo9cMC050rglATFWXpsxbCLTWc+IE/wK2SES+1fXN63Wm4J2/hIXAYL+krWmwn4hsEZGlInKFiAQz3SCzy67zf5avpXyQafR3oKrL8HpnTej+5vUYTc9JFbAMmCIixcBQml87WrsO93WrRGStiNzj9zjBzlH6/L+3HwIPikgOcA9wn/VQy4gHgVkiUgTg/1t3CnB/JhvV36hqstcvMAxYDjyc4Wb1GHbN6DHsetFDdPSaYcGJzpUH7GwybwfQ0hPOPwOTgYHAecBPROTUrm1er5OHd/4SEt/3+SfGHfAKsBcwCK/XzqnApRltkdlVP8QbujQcuB14SkTG0fzvAFq/rvQXbZ2TvJTppsv6ky3AAXjDNqbhvf+H/GV2jjpAVe8APgHexAvq/F9mW9Q/qep6vH/zTvZnzQK2qOrbmWtV/yUiAeCPwEuqelum29OT2DUj8+x60fOke82w4ETnqgQKmswrACqarqiqi1V1narGVfW/wM3AV7qhjb1J0/OZ+L7Z+eyvVHW5qq5QVVdV3weuxn6PeiVVfVNVK1S1TlXvA14DjqcD15V+pK1zUpky3XRZv+EPLZyvqjFV3QhcAHzOHw5k56jj7sALBP9WVesy3Zh+7D7gDP/7M4AHMtiW/u6neAHN72a6IT2UXTMyz64XPUta1wwLTnSupUBQRManzJsKLEpjWwWkS1rVey3CO38JU4GNqmp5GFpnv0d9R+Jn2ejvQLyqPhG8601/1fSc5ALj8PJQlAPraX7tSOc63Jep/zVg56hjRCQPuAm4C5jXX6rl9FCPA/uIyF7AF2joDWS6kYicgtdT8yuqGs10e3oau2b0GI9j14seoSPXDAtOdCJ/3PNjwNUikisinwFOpIVInYicKCLF4jkQL4r0RPe2ODNEJCgiWYADOCKS1UqehPuBb4jInv6YsR8D93ZfSzMn3XMkIseJyGD/+0l4yRL7xe9RXyIiRSJybOLnLCKnA4cDz+H9Y3qCiBzm34RfDTymqn3+KXcbfwd/A/YSkZP85T8B3lPVD/1N7wd+7F9jJ+ENnbs3A2+hy7V2jkTkIBGZKCIBP0/Pb/C6UiaGcvSbc9QJbgbmq+q5wDPAHzLcnn7LT4D7V7yuwW+p6uoMN6nfEZH9gN8Cs1V1c6bb00PZNaMHsOtFz9Dha4aq2qsTX3hl7R4HqoDVwGn+/MOAypT1HsbLrl4JfAh8N9Nt78ZzNA/vKV7qax4wyj8fo1LW/R6wES+Xxz1AJNPt70nnCPiFf36q8BLMXA2EMt1+e3X45z0Q+B9et/rtwBvAMSnLT/OvJ1V4waeSTLe5m85Li38H/rKj/WtnDfASUJayXQSvtOZO/+/je5l+L919jvCeUKzwf2fW4wUjhvTHc7Sb5/dE4NPE3xxevo5PgNMz3bb++gJm+L/n52S6Lf3x5V9fYv5nkcTr2Uy3q6e87JrRs152vcj8q6PXDPE3MsYYY4wxpkcTkVF4gckhqto0CbkxxiTZ9aL3sWEdxhhjjDGmx/OzvX8P+JPdaBhj2mLXi96ppXH+xhhjjDHG9Bh+zp2NwCq8soDGGNMiu170XjaswxhjjDHGGGOMMRllwzqMMcYYY4wxxhiTURacMMYYY4wxxhhjTEZZcMIYY4wxxhhjjDEZZcEJY4wxxhhjjDHGZJQFJ4wxxhhjjDHGGJNRFpwwxhhjjDHGGGNMRllwwhhjjDHGGGOMMRllwQljjDHGGGOMMcZklAUnjDHGGGOMMcYYk1EWnDDGGGOMMcYYY0xGWXDCGGOMMcYYY4wxGWXBCWOMMcYYY4wxxmSUBSeMMcYYY4wxxhiTURacMMYYY4wxxhhjTEZZcMIYY4wxPY6IXCAi80WkTkTubWfdi0Vkg4jsFJG7RSSSsqxMRF4UkWoR+VBEju7yxhtjjDGmwyw4YYwxxpieaB1wLXB3WyuJyLHAZcBRwGhgLHBVyioPA+8ApcD/AX8VkYFd0WBjjDHG7DpR1bZXEJkJzAb2B0qAbXj/yD+uqi92bfOMMcYY05+JyLXACFWd08ryPwIrVfVyf/oo4CFVHSIiE4D3gQGqWuEvf9Vf/odueQPGGGOMSUurPSdE5EgRWQjcDxQCjwM3+V/zgXtFZKGIHNn1zTTGGGOMadEUYGHK9EJgsIiU+suWJwITKcundGP7jDHGGJOGYBvLfgpcCjyvLXSvEBEBjgGuAWZ0TfNaN2vWLH3uuee6+7DG9DaS6Qb0FHbNMKZdvfV6kQfsSJlOfJ/fwrLE8uEt7UhE5gJzAfbcc89pixYtavGA/133Jm9vfcPbpmHrJtMNE9Lk1DbaRlqYhzbeRto7TgvzWjquNF+raVua/xI03otI5n5N4m6cumiQb+79jYy1wTTSW68Znc4+YxiTlnavGa0GJ1T10LY29AMW//Rf3W7Lli2ZOKwxppeya4YxfVYlUJAynfi+ooVlieUVtEBVbwduB5g+fXqr414PHXYQhw47aFfb2ycknlspKV8VVF0UcNXF1TgKKBDXOLiK4i2P4/r7cXHVn6+KquKCN+0qihLXOM+t+Tu54TBZ4TjXzf8FF039Dtmh7Ey8dWOasc8YxnSOtnpOGGOMMcb0dIuAqcCf/empwEZV3Soii4CxIpKfMrRjKvDHDLSzT5Fkb43GPTzA6ZLjnV/wTe778H4cJ8aw/EJ+ufAmzppwFqMKWuwEY4wxphdKq1qHiOSKyI9E5FER+Wfqq6sbaIwxxpj+R0SCIpKFd7friEiWiLT0UOV+4BsisqeIFAE/Bu4FUNWlwLvAlf72XwL2AR7thrdgOlFAAsyZdDZFoUGICKMKBvKnTx7k5bWvZ7ppxhhjOkm6pUTvB84APgFea/IyxhhjjOlsPwZq8MqEnuF//2MRGSUilSIyCkBVnwNuAF4EVgOrgCtT9nMKMB0oB34OfEVVN3fbuzCdRkT44pgvMqFgCqrKkLxSFmz9L/csejjTTTPGGNMJ0h3WcRRQpqrbu7AtxhhjjDEAqOo8YF4ri/OarPsr4Fet7GclMLPzWmYy7eAhhzAweyCvbniR0uxCKuo2cdUbN3D5ARcTckKZbp4xxphdlG5wYg3QamKo9ixYsODYYDB4paoOIf3eGm264YYbWLJkSWfsypheKRQKMWjQIAoKmuZ6M8YYY/q2cYV7kB/K4+9rniY/kosTcLj8v1fx/f0vZHDuwEw3zxhjzC5INzhxEXCbiNwAbEhdoKrr2tpwwYIFx0Yikd+VlZXVZ2dnlwcCgV0OcqRavHjx6MmTJ3fGrozpdVSVmpoaPv30UwALUBhjjOl3BuUM4ctjTubJlX8jJ5TFhNJh/OKdX/PVPb7GAUOmZrp5xhhjOijdXgwKHAb8D68XxRpgrf+1TcFg8MqysrL63Nzcms4KTBjT34kIOTk5DB8+nE2bNmW6OcYYY0xGFIQLOWnsV8kJ5hEJhpk0YBR/W/EXHvnoiUw3zRhjTAelG5y4DS/z9V7AWP81xv/aJlUdkp2dXburDTTGtC47O5toNJrpZhhjjDEZkx3M4Yujv8yArMGEAkHGF4/iwx3vct3/bibmxjLdPGOMMWlKNzgxGPixqi5R1VWpr3SOYT0mjOkaiTrzxhhjTH8WdsIcO/J4yvLG4gQClBUOw5UKLnvtKsprt2e6ecYYY9KQbnDiBWBaZxywuro68vbbb+//ySefjEnM27x5c8nChQv3fvvtt/f76KOPxkWjUaczjmWM6f1EZLyI1IrIgynzThORVSJSJSKPi0hJJttojDEm8xxxOGzokexZvDciwsiCIQzMzWHemz9j0ZaPMt08Y4wx7Ug3OLECeEZEfisil6e+OnrA1atXj8rOzq5KTFdVVWWtWbNmdFlZ2YqpU6cuDAQC7sqVK0d3dL+msZUrVyIirF27ttP3fdlll3HFFVd0+n5Tbd68mdGjR7Nly5YuPY7pFW7By3cDgIhMwRtqdiZer65q4NbMNM0YY0xPIiJMH3gQ0wYcCMDQvIGMLhzInUvu5Mnl/8xw64wxxrQl3eDE/sBivJwTx6S8ju7IwTZv3lzsOE48Pz+/IjFvy5Ytpfn5+dsLCwsrg8GgO2LEiHU7d+4sisVinVJytCNmzpyJiPDnP/+50fw333wTEaGsrKy7m9TjrF69mjvvvJNLL7007W3mzJnDueee26HjDBw4kNNOO42rrrqqo000fYiInAJsB/6VMvt04ClVfUVVK4ErgC+LSH4GmmiMMaYHmlKyDzOGzEQQBuQUM6ZwGK+s+xe/fuc2XNfNdPNMNxCRB0VkvYjsFJGlInJuyrKjRORDEakWkRdFZHTKsoiI3O1vt0FEvpeZd2BM/5NWAEBVj2zl9dl0DxSLxQLr168fPmrUqEYVPmpra7Oys7NrEtPZ2dl1IqI1NTVZTfexYcOGAR988MHkDz74YHIs1jUJjiZPnswdd9zRaN4dd9xBfytb2lqSxd///veceOKJ3VK68utf/zr33HMPO3fu7PJjmZ5HRAqAq4GmHwqmAAsTE6q6DKgHJrSwj7kiMl9E5m/evLkrm2uMMaaHGVuwB0cNn0VQghRl5TO2aDgbalZy+evXUBmtan8Hpre7DihT1QLgi8C1IjJNRAYAj+E93CgB5gOPpGw3DxgPjAaOBH4gIrO6s+HG9Ffd1jthzZo1w0tKSrZEIpFGd72u6zrBYDDeqFGBQDwejzfLOzFkyJAte+2115K99tprSTAY7JJ2fvnLX+add95h+fLlAFRUVPDoo49yzjnnNFqvurqaCy+8kJEjRzJgwABmz57N6tWrk8tnzpzJJZdcwkknnUR+fj7jxo3jiScal7X6/e9/z8SJEyksLOTggw/m1VdfTS6bN28eRx11FBdffDGlpaWMGDGCn//85422f/nllznooIMoLCxk0qRJ3Hbbba2+r4ULF3LEEUcwYMAAiouLOe6441i2bFly+Zw5czj99NOZM2cOJSUlfPe7321xP48//jjHHHNMo3kiwk033cS+++5Lfn4+Rx55JJ988gkAN9xwAw899BD33XcfeXl55OXlEY/H03p/48ePZ8CAAbzwwgutvi/Tp10D3KWqTccm5QE7mszbATTrOaGqt6vqdFWdPnDgwC5qpjHGmJ5qWO5wjh35BbKcbPLCOYwpHI4E6vi/169m+Y6VmW6e6UKqukhV6xKT/msc8GVgkar+RVVr8YIRU0Vkkr/u2cA1qlquqkuAO4A53dp4Y/qptO7wRSSK9wfdjKqG29u+srIyu7KysmDKlCmL93/w5IbEmm+QSGKXD4xqtNEHzZ+CNrOg3TVYcMZf2l8pRVZWFqeffjp33XUXP/3pT3n44Yc54ogjGDp0aKP1Lr74Yt59913eeOMNioqKuPDCCznhhBNYsGABjuPFVe677z6efPJJ/vKXv3DzzTdz9tlns27dOnJycnj44Ye54ooreOaZZ5g2bRr33Xcfs2bNYvHixYwe7fUse+WVVzjmmGNYv34977//PscddxyjRo3itNNOY8WKFcyaNYvf//73nHHGGcyfP5/jjz+ekpISTj755GbvS0SYN28ehx56KLW1tZx77rmcccYZvP7668l1/vKXv/DAAw9w1113UVdX12wfNTU1fPjhh+y5557Nlt1+++08+eSTDB8+nEsvvZQvfvGLvP/++/zgBz9g8eLFBINB7rzzzkbbtPX+Evbee28WLFjAl7/85Q78FE1vJyL74g0b26+FxZVA0647BUBFC+saY4zp50qzBnDcqC/ywtpnARhbOIKVO9fxm4W38IWyL3L0qMMy3ELTVUTkVrzAQjbwDvB34Kc07oFZJSLLgCkishEYmrrc/352C/ueC8wFGDVqVNPFxphdkG7PiaNpnGtiDt4f6kXpbLxz5878+vr68HvvvbfPLrSx25133nncc889xGIxbr/9ds4777xGy13X5b777uPaa69l+PDh5ObmctNNN7FkyRLeeuut5Hpf+9rXOPTQQwkEAsydO5cdO3bw8ccfA3DPPfdw/vnnc9BBBxEMBvnGN77BPvvswx//+Mfk9kOHDuWHP/wh4XCYadOmMXfuXO69914AHn74Yfbff3/mzJlDMBjk4IMP5vzzz28WAEjYZ599OPLII4lEIhQWFnLllVfyxhtvUF1dnVxnxowZfO1rX8NxHHJycprto7y8HKDFIR2XXHIJe+yxB9nZ2dxwww0sW7aMN998s83z3Nb7SygoKGDbtm1t7sf0STOBMmC1iGwAvg+cJCILgEXA1MSKIjIWiABLu7+ZxhhjeoP8UD7HjTyBAVkDiQTDjCsaSUEkm2dXP8Uf3rs/080zXURVv433EPQwvKEcdbTdAzMvZbrpsqb7tt6ZxnSytHpOqOrLTeeJyH+BP5FGlvzBgwdvGTBgwDaAN/Z8iHXr1g2pr68Pl5WVrY5Go8GPPvpo8rhx4z7Oy8urXrFixWhVlfHjxy9va5+LFy+e1tIT/M6w1157MXr0aK655ho2bdrErFmzePjhh5PLN2/eTF1dHWPGJKuhkpeXx6BBg1izZg2HHHIIQKPeFrm5uYA3TARgzZo1fPWrX2103HHjxrFmTUNKjtGjRyMiyemysjIee+yx5Papx09s33ToSMKyZcu49NJLefPNN6moqEjuN1EVI7H/thQXFwO0mAMidducnBwGDhzYbqWQtt5fws6dO5u9T9Mv3I53fUn4Pl6w4lvAIOB1ETmM/8/enQdGVV4PH/+eO2sm+wYJkACCAoIigqgg7la0uLdVccNdu9lWrdZfVSx2tW+rrda6L7W11t261V0RFQQFFFCQHRIgAbJvszzvH/fOZLLBAEkmCefTjpm52zxzydzce+55zmPnT/0KeM4Yo5kTSimlOuR3p3DCoJP5oPQdNtauZ5+sItZVlbKyegk3f/wb/u+Qa/G7fclupupkxpgw8KGInI99HrGjDMyauNcNreYppbrYntSc2AgkFB1wuVwRr9cbij5cLlfEsizj9XpDqampDUVFRWvXrFkzdOHChWPD4bA1ZMiQtXvQrk5xxRVXMGvWLC655JJYN42o/Px8fD4fa9asiU2rqalhy5YtFBUVJbT9oqKiFusDrFq1qsX6a9euxZjm3jRr1qxh0KBBCa8f76qrriI9PZ3FixdTVVXFnDlzAFps37J2/OuQkpLCiBEjWLp0aZt58W2pq6ujrKws1taOtrujzxf15ZdfMm5ce5n9qi8zxtQZYzZFH9gnCw3GmDJjzBLgKuCfwBbsuxnfT2JzlVJK9RIey8MxA05gWMZ+WCIMzhhAli+dhkglN340k401pcluouo6buyaE60zMFOj040x24HS+PnO8yXd2E6l9loJBSdEZFKrxwnAI8Cy3XnToqKikuHDh6+Ovs7Pz982duzYL8aPH//5iBEjVno8nvCO1u8O5557Lm+88QbXXHNNm3mWZXHhhRdy8803U1JSQl1dHddeey0jR45k4sSJCW1/xowZ3HfffcybN49QKMQjjzzCwoULW9RbKC0t5Y477iAYDPL555/zwAMPcNFFF8Xat2DBAh5//HFCoRDz5s3jvvvu49JLL233/aqqqkhNTSUrK4vy8nJuueWW3dgrcPrpp7dboPLPf/4zK1eupKGhgRtvvJF99tmHQw89FICCggJWrVrVZuiuHX0+gG+++YaysjKOP36XRqxVfZAxZqYx5vy41/8yxhQbY1KNMacZY7Tvj1JKqYRYYjGp/xQOyDkIESjKKCDXn4XHFeEPn/2Jj0o+TXYT1R4SkX4ico6IpImIS0ROBM7FHpr8eWCMiJwlIn7gFmCxMeYrZ/XHgV+KSLZTJPNy4NEkfAyl9jqJZk582OrxLDAQuKSL2pV0fr+f448/PtaVobU///nPTJgwgUMOOYTi4mJKS0t56aWX2mRZdGT69OnceuutnH/++eTm5nLvvffy6quvxrpYAEyZMoXS0lIKCgqYNm0a11xzTSx4MXToUF599VXuvvtucnNzueCCC5g1a1abriLx7Z09ezYZGRlMmTKFadOm7eIesV199dW88MILbbp2XHbZZZx55pnk5+ezaNEiXnzxxdi+uOyyy6itrSU3N5esrCzC4fBOPx/Aww8/zIwZM8jMzNyttiqllFJKtUdEGJc3gYn9JgEwID2fgtR8vC6LZ1b+h8eX7VpBddXjGOwuHBuA7cAfgZ8YY14yxpQBZ2EXxtwOHAqcE7furcBKYC3wPnCHMeb1bmy7UnstiU+r7wqLFi1aM3bs2PLO3m5X1pzoCWbOnMmHH37YI4fRvPHGG/F4PMyaNQuw/8DPnj2bI444IuFt7OzzlZWVMWHCBObPn48WGdqxZcuWMWrUqI5mS0cz9jYTJkww8+fPT3YzlOrJ9HgRR48Ze4+11auZvek9IiZMVWM966o2YjBkuPtx44Qf43F5kt3EnkqPGQ49XiiVkJ0eMxIqiKlUvN/97ndd/h75+fmsXZv00iNKKaWU6uMGpw/F70rh3ZI3yPDByJxhfL1tJVWhLdwwZya/mPAz8gO5yW7mXsUZUvxgIAfYBnxujPk8qY1SSnW5Drt1iMhtToGYDjn9uG7r/GYppZRSSinVPfoHCphadAoBdypul3BA/igEC5crxO2f/o7Pt3yZ7Cb2eSLiEZGfichq4GPgJ8Dpzs+PRGS1iPxURDSVRak+akeZEz5gtYi8CLwJLAWqsIfT2R84HvuA8WAXt3GvNHPmzGQ3IWG70zWoN30+pZRSSvV9Wb5sTio6hbc2vk5lUwXj+o9i4ZbleN1BHv/6MVZWHsV39t29ml0qIV8AC7ALUH5gjGmKzhARL3AkcDGwGOiwP6tSqvfqMHPCGHMjMAEoA2ZiHwhWOz9vA7YCE4wxN3V9M5VSSim1NxGRHBF5XkRqRWStiEzvYLnXRKQm7tEkIl/EzV8jIvVx89/ovk+heptUTxpTi06hX0oBwUgTB+XvS8Cdgduy+GTzB/xhwd1tRh9TneZMY8x5xpi34gMTAMaYJmf6ecCZSWqfUqqL7XC0DmPMOmPMTcaY/YEAMAgIGGNGGWN+YYxZ1y2tVEoppdTe5h6gCegPnAfcKyKjWy9kjDnJGJMWfQAfAa2HWjglbplvdXnLVa/mc/k4fuBUitMGEzRB9s0ZwMDUYkSEsob13PDRTCobq3a+IbVLjDFLE1xuWVe3RSmVHIkOJYoxpsEYU2qMaejKBimllFJq7+bUvDoLuNkYU2OM+RB4CbhgJ+sNAaYAj3d5I1Wf5rbcHFl4HCMyRxExYXICPsbljyMciYA0cssnt/PVtm+S3cw+T0RSReR2EXlZRO4UkX7JbpNSquskHJxQSimllOom+wEhY8zyuGmLgDaZE61cCMw2xqxpNf2fIlImIm+IyNiOVhaRK0RkvojMLysr262Gq77DEouJ/SZxUO54AELUcHzx0TSFI3jdwt+/vI9XV7+T5Fb2eX/BHn7wr87rfyexLUqpLqbBCaWUUkr1NGnYRbjjVQLpO1nvQuDRVtPOA4YAg4F3gf+JSFZ7Kxtj7jfGTDDGTMjPz9/FJqu+SEQ4MHcch/efgiCUN5Zy6j4nEI648Lgs3t74On9d+KDWoegkIvLjVpOGG2P+zxjzP+Ba7OFFlVJ9lAYn+qg1a9YgImzYsKHTt33jjTdy8803d/p2u9OSJUsYMWIEjY2NyW6KUkqptmqwRweLlwFUd7SCiBwBFADPxE83xswxxtQbY+qMMb8FKrC7fiiVsH0zR3DMgBNwiYuNdes4aegUMtx5WCKsq13B/338a+qC9cluZl8wSkQ+EJH9nNeficgjInI58CTwfhLbppTqYhqciHP00UcjIvznP/9pMX3u3LmICEOGDElOw3qQdevW8eCDD3L99dcnuyl7ZPTo0Rx88MHcfffdyW6KUkqptpYDbhHZN27aWGDJDta5CHjOGFOzk20b7DRxpXbJoLRivjXo2/gsH6V1Gzm4YD9GZx1EOGIIUcMvPp7J6sr1yW5mr2aMuRq4BXhWRG4AfgF8DBwEfAK0O2qPUqpvSCg4ISI/iPbRFJHxzpBeK0VkQtc2r/uNGjWKBx54oMW0Bx54gFGj9q7hlIPBYLvT7733Xk477TQyMlrf0Oo+HbVtV11yySX89a9/1VRMpZTqYYwxtcBzwK+cgniTgdOAf7S3vIikAN+jVZcOESkWkcki4hURv4hcD+QBc7r0A6g+Kz+lHycVn0qaO42tjWVkBlxMG3IKwXAErwvuXPQX3tvwUbKb2asZY94DDgHygQ+AucaYHxhj/uQcG5RSfVSimRPXAhud57/GLkbzCPD/uqJRyXTmmWfy+eefs2rVKgCqq6t59tlnufjii1ssV1dXxzXXXENRURF5eXmcfvrprFvXPLLq0UcfzbXXXstZZ51Feno6w4YN48UXX2yxjXvvvZcRI0aQmZnJYYcdxuzZs2PzZs6cyXHHHcdPf/pTcnNzGTRoEL/73e9arP/+++9z6KGHkpmZyciRI7nvvvs6/FyLFi3iqKOOIi8vj+zsbE466SRWrlwZmz9jxgzOO+88ZsyYQU5ODj/+cesuf7YXXniBE044ocW0rVu3cumll1JUVER+fj7f+9732Lx5c2z+kCFD+M1vfsNxxx1HWloaY8aM4aOPWv7hfuCBBxgzZgyZmZmMGzeON95oHoZ+5syZHHvssVx33XX079+fU089FYCHHnqIYcOGkZGRwQUXXMD555/PjBkzADj77LO55pprWrzHww8/zPDhwzHGAHDkkUeyadMmFi5c2OF+U0oplTgR+Unc8+F7uLnvAynAFux07quNMUtEZIqItM6OOB27u8a7raanA/cC27HPY6YCJxljtu5h29ReLMObydTiU8n25VIdrKK8aR1XjLmYYFjwuixeWv08D375z2Q3s9cSkXxgDHA78CPgURH5lYh4ktsypVRXcye4XK4xplxEfMDh2CcBQeBnu/qG18+5YfyurtOhBO573DH597u0Sb/fz3nnncdDDz3Er3/9a5588kmOOuooCgsLWyz305/+lIULF/LJJ5+QlZXFNddcwymnnMJnn32Gy+UC4LHHHuOll17i6aef5q677uKiiy6ipKSEQCDAk08+yc0338wrr7zC+PHjeeyxx5g6dSpLly5l8ODBAHzwwQeccMIJlJaW8sUXX3DSSSdRXFzM9OnTWb16NVOnTuXee+/l/PPPZ/78+Zx88snk5OTw3e9+t83nEhFmzpzJpEmTaGho4LLLLuP888/n448/ji3z9NNP849//IOHHnqo3VoM9fX1fPXVV+y///6xacYYTj/9dEaMGMGXX36Jx+PhRz/6EdOnT+ftt9+OLffwww/z4osvMnLkSK677jouuugiVqxYAdiBid///vc8++yzHHDAAbz++uuceeaZLFy4kOHDh8f2xbe//W3Wr19PKBTigw8+4Ic//CGvvPIKRx55JE8//TQXXXQR06fb2X5XXnkl3/3ud/nDH/6Az+cD4MEHH+Syyy5DxM7m9fl87Lvvvnz22WccfLDWV1JKqU5wG3Cn8/wz2taNSJgxZhv2+Ubr6bOxC2bGT3sSO4DRetklwIG72walOhJwBzhx0Ld5r/QtNtWVsHDrp/zkoCt4cMmTNFLF15WLufWTDdx0yE/xubzJbm6vISJXYQclVmAXsb0KmAjcDMwTkauMMXOT2ESlVBdKNHOi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ey1nnXUW6enpDBs2jBdffLHFNu69915GjBhBZmYmhx12GLNnz47NmzlzJscddxw//elPyc3NZdCgQfzud79rsf7777/PoYceSmZmJiNHjtzhBeyiRYs46qijyMvLi13wrly5MjZ/xowZnHfeecyYMYOcnBx+/OMft7udF154gRNOOKHFNBHhzjvv5KCDDiI9PZ1jjjmGb775JjY/FArxm9/8hv3224+srCwmT57M/PnzY/PffvttDj30ULKzs8nPz+ecc85hy5Yt7b5/OBzm6quvZuLEiWzevLnDz/vKK6/Qr1+/FkGWmpoa0tLSeP99e2TctWvXctppp5GXl0dRURE/+clPqK+vb3d7Y8faf6NGjBhBWloas2bNAuCmm25in332IS0tjWHDhnHnnXe2WG/u3LmMHz+e9PR0jjjiCH71q1+1yMCpq6vjuuuuY+jQoeTk5DB16tQW+w7ghBNO4IUXXujwsyqlVE9kjHkWO1OiDpiAPSzwCcaYZ5LaMKV6KLfl5sjC4xiROYqICfN+6dt8VbEUy7L40djLOHbgtwiFI7hdYX49//cs2Lw42U3ubMfGPT+hg0frbKwduRcYBZxijIk/wXseGCMiZ4mIn7ZDkj8O/FJEsp3RCi/HHhhAKdXFEh5KNMoYY7D7be2Wx5c/OH53123BDfOXf7TTxS7c77Jd2qzf7+e8887joYce4te//jVPPvkkRx11FIWFhS2W++lPf8rChQv55JNPyMrK4pprruGUU07hs88+w+VyAfDYY4/x0ksv8fTTT3PXXXdx0UUXUVJSQiAQ4Mknn+Tmm2/mlVdeYfz48Tz22GNMnTqVpUuXMniw3R33gw8+4IQTTqC0tJQvvviCk046ieLiYqZPn87q1auZOnUq9957L+effz7z58/n5JNPJicnh+9+97ttPpeIMHPmTCZNmkRDQwOXXXYZ559/Ph9//HFsmaeffpp//OMfPPTQQzQ2NrbZRn19PV999RX7779/m3n3338/L730EgMHDuT666/n1FNP5YsvvsDlcnHrrbfy1ltv8frrrzN48GAeffRRpk6dyooVK8jOzsbn83H33Xczbtw4ysvL+d73vsc111zDk08+2eI9qqurOfvss/F6vbz33nsEAoEO/x2nTp2K2+3mlVde4fTTT499voKCAo488khCoRDf/va3mTx5MmvXrqWiooLT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5iS5nUqpLpLoaB17nXPPPZc33niDa665ps08y7K48MILufnmmykpKaGuro5rr72WkSNHMnHixIS2P2PGDO677z7mzZtHKBTikUceYeHChbG77gClpaXccccdBINBPv/8cx544AEuuuiiWPsWLFjA448/TigUYt68edx3331ceuml7b5fVVUVqampZGVlUV5ezi233LIbewVOP/103nrrrTbT//znP7Ny5UoaGhq48cYb2WeffTj00EMREa655hquu+66WJeWmpoa/ve//1FSUhJrW2ZmJunp6axbt65N4c+oO+64g/PPP58jjjiC5csT++N78cUX8/DDD/P444+36FIxceJEhg8fzrXXXktdXR0lJSXcfPPNsW4freXn52NZVuwzRNvtcrnIz89HRHjllVd47bXmAWymTZtGdXU1f/rTnwgGgyxcuLBFN5N+/foxffp0vv/978eGCq2oqOD5559vkTny5ptvxupmKKVURyImQjASoikcpC7YQFVTDRWN1VQ11VDTVLtLXdY6yTfAia2mHQ+sbGfZ1tKA1ml6lUB6O8v+B7ueRD524bpbRCQa4U1z1ktkO4jIFSIyX0Tml5WVJdBMpbpefko/Tio+lTR3GuUNZby2/r9UN9lfj+/sewoXjLiIpnAEr1u4f8kD/HfVm0lu8e4xxvwdGAo8jZ319CDwpvPzGOAZYKgx5v5ktVEp1bV2VHNitYis2tmjOxvbnfx+P8cff3ysK0Nrf/7zn5kwYQKHHHIIxcXFlJaW8tJLL7XJsujI9OnTufXWWzn//PPJzc3l3nvv5dVXX22R9j9lyhRKS0spKChg2rRpXHPNNbHgxdChQ3n11Ve5++67yc3N5YILLmDWrFltuorEt3f27NlkZGQwZcoUpk2btot7xHb11VfzwgsvtOnacdlll3HmmWeSn5/PokWLePHFF2P74rbbbuO0006LDUG677778ve//51IJALYxTQffPBB0tPTOfPMM9st6Bl1yy238LOf/YyjjjqKxYt3XqX6nHPOYdWqVdTU1HDaaafFprvdbl5++WU2bNhAcXExEydO5NBDD+WPf/xju9tJSUlh1qxZnHvuuWRlZfHrX/+aE088kQsvvJCJEyeSl5fHM888wxlnnBFbJysri1deeYV//vOfZGdn88Mf/pAZM2bg8/liyzzwwAOMGDGCo48+mvT0dA444ACefvrpWICkoqKCV199lauuumqnn1UptXeImAihSJigE4SobqqloqGKysZqapvqqAs2EIyEEASvy4PX5QXaBl27wW+Bp0Tk9yJyqYj8DjuQkEh30BqgdYGjDKC69YLGmKXGmBJjTNgpwHkX8J1d3Y6zrfuNMROMMRPay6JTKlkyvJlMLT6VbF8u1cEqXlv/X7Y22MOJjus3hl9OuJFIxI3HZfFeyZvc+fn9sfOs3sQYU+d8D082xhQYY3zOz5OMMfcZY1pnYyml+hDp6E6KiJwX93If4PvY1a5XY0c1LwH+Zoy5fUdvsGjRojVjx47t9MGYu7LmRE8wc+ZMPvzww3azFJLtxhtvxOPxxIbTFBFmz57NEUcckeSW9Xy/+MUvWLBgAW+88UbCy7tcLm6/veOv2bJlyxg1alRHs5NyRdITTZgwwcQPYatUT2eMIWwiGBMh7AQkwpEwRgzY/8cSwRILS3acCBkMh8j0pbWbGRan048XInIS8ENgCLAGuMcY82oC66UC24HRxpgVzrTHgRJjzA5rTojIDcChxpgzReRbwMNAUbRrh4isxa7Yv8OaE3rMUD1RU7iJ90rfYlNdCW7xcPSA4xmQOhCAUCTEb+ffRVXQHo7dRSo3T7yOVM+Oi4jvAT3HcOjxQqmE7PSY0WFBzGg1WwAR+QB7jOD5cdOeA+4EdhicUH1PR90uVFtvvPEGBxxwAP379+fDDz/k/vvv7zA7oz2//e1vu7B1SqmewBhDxESIYNeHiNaIiGDfPBDsILBg4bJcOwsw9Agi4sbOYLjWGPPazpZvzRhT65xn/EpELgMOAk4DJrVeVkROAz4AKoBDgB8D0dHE3gPCwI9F5O/Y3T4A3tnVNinVE3hdXo4beCJzNn3AmuqVvLPxf0wqOJJ9MobjttzcPPFaHl/2NAvLPwWrlps+uo1rDvo++2QO3vnGlVIqyRKtOXEQsLDVtMXOdKWS4je/+Q1paWntPmbPnp3s5gF2Mctx48aRlpbGJZdcwvXXXx+rG6KU2rtEi1MGIyEaQ03UNtVT2VhDZWMNVU211DTV0hBqImwMLsvldMnw4HF5cFtuXJbVKwITAMaYEHAO0LgHm/k+dlHNLcCTwNXGmCUiMkVE4od0Oge7vkU18Djwe2PMY047moDTgQuxgxeXAKc705XqlVziYkrB0eyffQARIny46T2WbGvu6nrhqO/ynWHfJRiO4HXDXxbdzdvrPkxii5VSKjEddutosZDIp8B/jDF3xE27DjjHGDNhR+tqtw6lupZ260iMplyq7hQxESJORkQoEiZkQoQjEcDYAQYjiEisW0ZXMcZgMDSFg+T4M7u1W4eIPAK8Yox5pjO32130mKF6gyXbFrOgfB4Ao7LGMCH/0Nj3fEN1Kf/v87/gdtnHoxGZB3DFARd05tvrOYZDjxdqb7WxZjP/WPpfrjn4fFLc/p0tvvvdOlr5AfCqiPwAWAsMxq6A/e0E11dKKaX6tGggoj7USMSEEbtDRiwI4XV5uuy9o0GIiInEuomETRjjdA0JhcMYY7o788IDPCEiV2HXm4hV5zPGXNGdDVGqrxqdcyAp7gAfbfqAZRVfUh+uZ3L/I3FZLgalF/KbSbfym0//TEOkghVVX3Lzx7/l/yb+DL/Lt/ONJ4nTLexF4CxjTEOy26OUaqusbhsPfvksz3/zNqFImPxADpeOOXOPt5tQcMIYM09E9gFOAQYCG4GXjTGth+dqTzgSiYhlWd0+hplSfV1vrMStVF8TioRpCgdpCgcBu0uG2/J2yXsZY4hgByDaC0JAtD6FnZERDUaEwuEuac9OBLG7YwC4nIdSqpPtkzEcvyuF90reYk31ShpC9Rw94Hi8Li8pbj+zDv8F933xOMsrv6QhUsEv5szkuoN/zMC0wmQ3vV3GmJCIjAdCyW6LUqql7Q1VPLrkBf6z/HUaw0EE4dtDj+TEwZM7ZfuJZk5gjKkC/rnTBdv6cO3atUcMGDCgxuv1BntLf1mlehr7ziiAIRKJEAqFKNtSRmpqapJbptTexxhDKBKiIdxEKBLGEsHdicUqdzcI0VM4dz6XAX81xtQnuz1K9XUDUgdyYtG3eXvj/9hUX8L/NrzCcQNPJOC2R+q48oALeWvdbF5e8xIeF/zhsz9xzvBzOHzA+CS3vEP/wB7p584kt0MpBVQ31fLEspf557KXqQvZCU3HFR/K1QeezT5ZRZ32PgkFJ0TEBfwCuAjoZ4zJFJETgaHGmL/vaN1QKHR5RUXF1dXV1TOMMTkkXoRzh7Zu3drjTsaU6gwmOkZg9Dn2hUrzfJvX4yE7K5u8vLxubqFSe69wJEyjkyVhiOAS9x5112gdhAibsN01o5cEITri3Pm8yRjzh2S3Ram9Ra4/j5OKT+WtDa+xvXErr697ieMGTSXTmwXA8cVTGJpRxF8X34vXZfH0yn/zTeUqLhj13eQ2vH0HA9fEdSmP7xb2raS1Sqm9TH2ogae+fp1Hl7xAVVMtAJMHjOP7Y89hVO4+nf5+iRbE/A1wPPA74GFjTJbTzeNZY8y4Tm9VArTwjOrN4ocOjEQiBCMhgpEgwXCICGFMxLk0EXCJM3yg03cdoD5YT25KDn73TvuM9o4rmW6gxwy1u1pmSYQQrF3OkogGHBIJQsR/1ztLQ7CRgtR8LGuH9wc6uyDm88Cdxpj3O3O73UWPGaq3agjV807JG5Q3lOGzfBw78ETyU/rF5tcEa7l93h8JUwdAhrs/N074EZ5dD7R22TmGiNza0TxjzG1d9b67S48Xqq9pCgd5bsVbPPTls2xtsCs5HNxvFD846FzG9euwEP/OdFpBzOnA4caYUhF50Jm2GhiSUCtEfMDfsAMcOcBK4BfRsc9F5DjgHqAYmAvMMMasTbBtSvVYrYMQTeEgYcKEwmHCJuQUrwOxBAvBsizcuBGr8y9OehM9ZqieIH7YTyMGCxde185rSdgjdUT6VCbEbloDvCgiz9C2IOZvktQmpfo8vzuFEwadzAel77Cxdj1vbHiFowqPY1BaMQBpnlR+c/jN/GXRA2yoXUlVaDM3zJnJTYdcS15KTpJbb+uJAQil9gahSJiXV73P/V88zaZae8DN/XOG8YODzuWwwgO7/Lwl0eBEAHuc8XheINEKum5gPXAUsA44GfiPiBwA1ADPAZcB/wVmAU8BhyW4baWSrnUQIhQJ0xQJ2tMiYSISIRI2IGCJHXhwWS47ENG3L052lx4zVFIYYwiZME2hJpoiwYSzJKIBiaDzvYe9KgjRkYOAz4FhziPKABqcUKoLeSwPxww4gY83f8jKquW8W/Imh/U/gn0zRwBgWRY/GXclL636H+9ufAu3K8Sv5v2WS0bN4KB+o5PcepuIFGHfIC3CPif4pzFmQ3JbpVTfFDER3lz7MX9f9BRrq0sBGJ5VxNVjz+HoQYd023lMosGJz4CLgQfjpk0H5iWysjGmFpgZN+llEVkNjAdygSXGmKcBRGQmUC4iI40xXyXYPqW6RdsgRIhgxLkz6hSsizh9yC0RXOJyuma4cLv37myIXaHHDNXdIiZCMByiIdSIwSBi7TRLIno8CEVChIxdVN4SC7eVcK3pPs0Yc0yy26DU3swSi0n9pxBwB/hi20I+3jyb+lAdB+QcFDsfOXWfExmWOZj7lzyMz23x2NePsrLyaM7a99tJbbuIHAG8DizGzp4cB9wsIicZY2YntXFK9SHGGD7YuIC/Lfw3KyrsJORBaf25euzZfGvwJFzWzgfaqmqqIt2T3inXOYmeQV0HvCci5wABEfkvMAHYrRMPEekP7AcsAa4GFkXnGWNqRWQlMBrQCw2VFPFBiHAkTDgSJmQiRCIRwoQxJkLE6TtuicR6UIkleOi8iv3KpscM1VWCkZCTJWEHFzzWzrOZwiZMKBIiHLGH5xQRDUh0QOydORH7zuc64FOTSLErpVSnEBHG5U0gxR1g3paPWLh1AXWhOib2OxxL7Bo0o3NHMnPiTfxm/v/DbTXx8eb3WVm1huvGXb2zOjVd6Q/Aj40xD8d9louBO9BMSaU6xdzSL7hn0ZN8Wb4CgP6BXC4/4DucMuxoPDs5rzHGsLpqDXNK5/Dl1iVcdcCVDM0YssdtSuhsyhjzpYiMAi7EPvlfC1xmjNm8q28oIh7sIUkfM8Z8JSJpQFmrxSqB9HbWvQK4AqC4uHhX31qpNtrLhIgGIewK+nZNCCMGcX4CCILLsrqkcJ1qSY8ZqrPFsiTCTURMGEtcOx1xI2IiTg2KoJNZsVd310iIk5L9X2AUdtfQfsAyETnVGLMuqY1Tai8zMmt/UlwpzN70Hssrl9EQruOIgmNigdVsfxa/mzSTOxbczdamEkprNxA0IXzsvM5OFxkFPNpq2uPAn7q/KUr1LYvKvuaehU8yf/MSAHL8mVwy5gzO2vcEfDvJGg2Gg3xe/jkfln5Eaa3d/cPCYmPNxu4LTgAYY8qA/7cnbyYiFva4xU3YYxeD3X88o9WiGUB1O224H7gf7Kq4e9IWtXfZURDCEIkVpmwdhIDmYpUiGojobnrMUJ0pFAnTFA7SGG5CwK77YnX8RzgakAhFQkSI7O31I3bHXcCnwGQnwykN+zziL8DpyWyYUnujwelD8btSeLfkDdbVrOWtja9xzIBv4XPZI3+5LBc3HnINz3zzMpMLDtnpRUoX24w9nGj8EBgH07YGnlIqQV9vW809i/7Nhxs/AyDdm8pF+5/GOSOmEvCk7HDd7Y3b+bj0E+ZunkddyB7pJ9WTymH9D+XwgsPI9GV2ShsTDk6IyOHYXTla3J1MtOK2k9r5ENAfONkYE3RmLQEuilsuFbtw1pJE26ZUVHuFKUMm3G4Qwh4mo3ldDUL0LHrMUJ0hOgxoXaiRiLFrwewoSyI6wkZ8YUuXuHCLdtvYDUcAg40x9QDGmBoR+Sn2yB1KqSToHyjgxKJpvL3xdbbUb+b19f/l+IFTSfWkxZb5zvBpSWxhzF3AqyJyH80jBF4J6CgeSu2i1ZUb+fvip3hz7ccApLh9TB/5bS7c/1TSvakdrmeMYVXVKj4smcOSbUtjI48NShvEEYWTGZt3YKd3a01oayJyO3bdiUXgDIrstJnEK27fi52idXz0RMXxPHCHiJwFvALcAizWwnZqR3YWhDAYIhGzkyCE3gHt4fSYoXZbNEuiKRzEEMFtuXFb7QclYoUtjV1HwmC0sGXnaAAygfjvbyZ2JpRSKkmyfTmcVHQqb218ncqmCl5b/xLHDZxKtq9nDCMKYIy5V0QqgBnAWdijdfzEGPNkMtulVG9SUrOF+xY/zSur3ydiDF7Lw3f3+xYXjzmDHH/HmQ5N4SY+K/ucOaUfsaluE2AX2D0odyyTB0yiOK24y66hEj3zuhKYaIxZvDtvIiKDnW00ApviPsyVxph/OhcZdwNPAHOBc3bnfVTfEx+EsO+AhgmZEKFwGIN9V1ODEH2PHjPU7ohmSTSEmwhFQk6AwYV0kPUQNuFYt41oQCKRqtQqYc8Dz4vI/2FnSwzBHvr32SS2SSkFpHrSmFp0Cu+WvMmW+k38b/3LHDPgBPoHCpPdNMQ+aN8FXKvBCKV2XVnddh788lme/+YtQpEwbnFxxr7HctmYs+ifmtvhetsatjGn9CM+3Tyf+rB9XyHNk8bhBYdxWMGhZHhb96rufIkGJ+qBpbv7JsaYtbS4bGwz/y1g5O5uX/V+exSEEMESDUL0JXrMULvCLlQZoiHUhCGCS9wdDgOqdSS61Y3AndgZTj7sYONjznSlVJL5XD6OHziVDze9y7qatby58XWmFBzN4PShSW2XMSbkjBD4w50urJSKqWis5tElL/Cfr1+nIdyEIJw8dApXHvg9itIL2l3HGMOKym+YU/oRy7Yti3XdKE4r4ogBkzkg94BuzSRN9J3+BPwSmNl1TVF7g0hsCM4I4ehFQlxhytbdMUQk9iXRIIRSCpwMCSfroTEcJGLCCB1nSdjHncheV0eiLthAWf02yuq2s7l2K+eN6vZ+5AdjDwd4FZCPPcrOPsA44KPuboxSqi235ebIwuP4dMvHfF25jPdL32ZieBIjs/ZPdtNewu7O8UyyG6JUT1fTVMcTy17mn1+9TG3Qzng4tuhQrh57NsOyitpdpzHcyIItnzGn9CO21Nt1Zl3iYmzeWI4onERRevvrdbVEz8yeBt4RkZ/QqkquMWa/zm6U6v3igxDBSJiwUw8ibOwAREdDdIIGIZRSbYUjYcImQjAcJBgJQ9xwnu2NuBGrIxEJETIhgD5VR6Ix3ERZ3fZY8GGL87OsfhtbnJ/RE5Sos/Y9gYB3x9W4O9l9wKnGGINz7uAc0+8DDujOhiilOmaJxcR+k0hxB1i4dQHztnxEfaiWg3InJPM8zAM8ISJXYXcLi0RnGGOu2NnKIvJD7HoVBwBPGmNmxM07DrgHKMbuGjrDydhERHzYNa++g11n7w/GGB2+VPVI9aFG/vP16zy65AUqm2oAOLxwLD846Fz2zx3W7jrl9eV8VPoxn26ZT0O4AYAMbwaHFxzGof0nku5Nb3e97pLoWdpTwAbs9My6HS+q9hbtFaUMEyYciWCcu5MRYyCa+2CMDtGplEpINDsiGA4RDAfjumC48Lg6/tMVNnaXjd5c2DIUCVFeX0FZ3Ta21G+nrG4bZfXb2eL8LKvbFjsJ2RGv5SE/kE2/lBxy/Jk0hoME6NbgRLExZlX8BGPMSqemjFKqBxERDswdR4o7wCebP2Rb4zYMBum4h2VXCwLRehMu57ErSoDbgROh+cAnInnAc8BlwH+x6+A8BRzmLDIT2BcYDBQA74rIUmPM67v1KZTqAsFwkOe+eZuHvniW8oYKAMb1G8UPDjqXg/uNarN8xERYUbGCD0s/4uvtX8ey0oekD2Fy4SQOyB3TY2puJXrGdhCQZ4xp6MK2qB6q3a4YkXCsPoTByYIgGogwLdYXEUQLU6oe7o21b5Ptz+aQ/gcnuyl7reixpSkSJBQJI9jHD5e14y4YbepISM+tIxE2EbbVV7QJNsQHIbY3VLU5jrbmFhd5KVnkB3LIT8kmP5BDv5Qc8gPZ5Kfk0C+QTYY3LbYPGoKNZPrSdrjNLlAmIsXGmHXRCU5gYlt3N0QplZh9M0eQ7kkn15+PJVZS2uAUxFwG/LXVaF0JM8Y852xrAjAobtaZwBJjzNPO/JlAuYiMdEb9ugg7k2I7sF1EHsDOwNDghEq6UCTMK6s/4P7FT1NaWwbAqJx9+MFB53J44dg25z0NoQbmb1nAR6UfUdZQDoBb3ByUP5bJhZMZlDaw2z/DziQanFgGZAOlXdgWlUTtFaSMVrK3u2EQK0xpDHa3DGkbhLCwYs+V6i2C4SCrar4ipcHLp1s+Yny/iRzSb0LSTsz2FhEn4NmcHWEfU1ziwuvawbCfRDDGOF09wrG7e5ZYSa8jEYyEWFO5kc11W9vtbrG1oSJW96IjFkJuShb5TrDBDjrYQYh+zs9sf0ZCv5/GGOpCdWyr305Ban5nfcxEPQ/8Q0SuBFZg3438G/ZdS6VUD1UQGJDU93cKYt5kjPlDF2x+NLAo7r1qRWQlMFpENgOF8fOd56e3tyERuQK4AqC4uLgLmqqULWIivLXuE/6+6CnWVJUAMCyziKvHns0xRRPbXHeV1Zcxp/Qj5m9ZQGO4EYBMbyaTCg9nYv9DSPN0+82KhCV6Fvco8KyI/BHYFD/DGKNFrXqJaADCQJuClMbJjDDiBCKMwS4GEc2FaE7t68l3JZXaHdsbKwmGw/jdhjSfn68rF/Plts8ZlrEfh/Q7hNQefBDvTYwxseNONDuCuCE83e0cUyLO8SliIoRMyMnWsoMYllhJPxYFIyG+3raGRWVfs6jsa5ZsXUljuGmH62T7MloEHfJSsunnZDzkB7LJ82ftNL0yYiJUN1VTE6zZ4aM2WBvbX7/OmYW3nfocXehW4GHs0b6i0exngJsTWVlEcoCHgG8B5cAvjDH/ame567Hvdg52lvubMeaOuPlrgP5A2Jn0kTHmW7vxeZRS3eddETnKGPN+J283Dbs4b7xKIN2ZF33del4bxpj7gfsBJkyYsON0N6V2gzGG2Rs/42+LnmT59rUADErrz1Vjz+bEwZNanCtETISvt3/Nh6UfsbxieWz6PhlDmVw4idG5o3FJ53bdaAg3UFq7kY216zkwdxwZ3sw93maiwYm/Oj9bV8w17HofsG4TLaAW61YAST+R7WqxE/m4DIhoyrN91DSx+QItMyKceQDRPRYNRKjkMcbQGG6korGC0trNbK4rY0t9GVMGTGJ8/7HJbl6f0C+Qx48O/DEfl3zK+6XvkhtIw+fysq52JWtXfUOurx+jsw+gKK0Yl+Xq08eQzhY9/jSFQ063CzvQ6bKsdrMjoiNrRIvoRpzjU/R4lOw+kcFIiOWxYMRylmz9hoZWwYii9AIGpvVrlflgBx/yUrI6zAoBu2ZGbbC2/WBDU/PzulDdTrt+RKW4U0h1p9IQbsDr7r7ghDGmFjjbKUw3BFhjjGl9UbAj9wBN2IGFg4BXRGSRMWZJq+UEuBBYDAwD3hCR9caYf8ctc4ozBLFSqndYA7woIs/QtiDmb/ZguzVARqtpGUC1My/6uqHVPKW61aebvuSehU+yuNwONPQL5HD5Ad/h1GHH4ImrpVUfqmf+lvl8VPox5Q1bAXsUnoPzxzG5cBIDUjsvE8oYw7bGrWysXc/G2vWUN5TFzkVy/XndF5wwxvTKq9OwiVDdVGuP/IDlXISL04/ZwuXcdbNEcIkFThDDkuYL854ovgZEJBqAID4AAXbKgzjZD4Idgog4mRNxRSmjBSnRIESy1AXr2FBTSmntZsoatlLRUEFNqIaGUD0h04RIBKud38VPN6docKKTHT7gEA4fcAhvrH6PDzd/QH5qBhneVLY1lTF78zv4yvwMSRvG8Iz9SPem4XZ5evSxIhmi2RGhSIimSJBwJIKdHeFqkx0R7aJhj+QTjnV3iBayFJGkd9MIRUIs376WhWVfs2jL1+0GI4rTCxmbP4KD+o3gwLz9yPa3Pu+1t1MbrKWsfssOMx3qQonXnA64A6R50tp/eJ2f7jRclouGYGPS0jidgMSuBCUQkVTsYQTHGGNqgA9F5CXgAuDGVtuPT/3+WkReBCYD8cEJpVTvchDwOXbAMX7YAQPsSXBiCXamFRA71gzDrkOxXURKgbHAm84iY511lOoWi8uW87dF/2bepi8AO9vykjFn8p39TsDnar7BsLluM3NKP2LBls9oijQ5y2ZzeMFhTOx/CKme1E5pT1O4idK6jWyoXU9J7Xrqw81lYCws+gcKGZRaxKDUzuna1LtKmO8GVweV3e00YUPYhAATG1WidRDDsiws7CCGywlkdHUQY0fdL+yTd7udOLUgEGK1lI0TgIj1aXbiEPEBCL2Q6h7BSIgtdVspqdnEprotbGvYRmVTFfWhOpoiDURMCMsCt9VBUEjAJfaTcCRCQyhE2IDgxiMe+gcKuvPj7FW+NfRoji6axIur/se8LZ/QLzUjdsH5ddUSllctpZ+/kCFpw+if0h+f24vH8mJZ1l4ZrGjOjogO82lrnR0RXS72s1VWRE84PrUIRpR9zZLylTQ4/TWjosGIsfn7MTZ/BNn+DIwxVAer2Vi7kcVbP6c6WE11U3UsCyL+j/mOCEKqJ5U0TxqpnlTSPemx1/HPUz2pnZ6e2cPsB4SMMcvjpi0CjtrRSmL/Ak3BHq403j9FxMK+2LneGLOozcpoH3KlegpjzDF7sr5TVNONM9KHiPiBEHYtnDtE5CzgFeAWYLFTDBPgceCXIjIfO2vrcuDiPWmLUolYvn0Nf1v4bz7YuACANE+AC/c/lekjTybgsQeciZgIy7Z9xZzSOayo/Ca27vDMYUwunMz+OaP2+GazMYaKpu2x7Igt9Ztb3NgOuAMMTC1iYGoRhYEBeDq5u2ifD050RERwJXASHM1SCJsQTU5fZxG7O0R8kMCyLFxYcRkZdhDDistMiD/p3ln3C+MMm2fA3paTBeESyylcGZfyHJfZa4l9cdTbhs7rTZrCQbY3VLGlfiultVsory9ne2OFfQESqiNsgkAYt8vC62rn4kHA7QKc4qHhSIRgxM5wcYkXn9tPujuNLF8WeSm5FAT6U5iaT7Y/I/bvWh+sJzclp9s+897I6/Zy1vBvc1zRFF5a9RqLyheS7U8lx59JujfA5oYSNjeUkOpKY2CgmEGpg0n1pOJ2u/GKB5e4+mywIjrMZzgSpjEcJGLCsRF53E63l2hWRHRYz2jhSmiuW5PsrAhoDkYsKlvOorKv+bL8m3aCEQUcmD8iFpDI8WfSFG5iU90mvq5YysbajZTUllAT7Hh4T0FaBBzisxxSPamke+1pAXdAs9hsaUBVq2kd9v2OMxP74PpI3LTzgM+w4/jXAP9zKvNXtF5Z+5Ar1XOIiAs4FCgyxjwlIgHAJDiCxy+x695EnQ/cZoyZ6QQm7gaeAOYC58QtdytwL7AWqAd+r8OIqq60pnIjf1/8H95Ya5dxTHH7OHfkyVw46lQynFG26kJ1zNv8KR+XfsK2RnvAK4/lYXz+wUwunERB6p7dsAxGgmyqK2Fj7Xo21K6nLlQbmycI/VMKYgGJLG92l57XJv/MsIezxGpOS+hAtEBb2BiME8QAA62CGAKIJUQikbh4gjNPQJwMDbsopRUrUhmMNHfFiBamFJEekfLcl4QiIbY3VFFeX8GW+q1sqS1na+M2KhsrqQ3V0hRuJEIIlwV+twtv64wcAZ8H7PNi++IiYgyRCIi48Vg+Aq4AGd4McvzZ9A/kMyC1gIJAPu52sntU8lmWRU5KFufsdwbH1h7J6+vf4quKJfjdHnL8mWT504Eallcv5Zuar+nvL6QoMIRsXy5ulwuv5bG7MlhuXOLqdTVcOh7Fx8ngigYZLG8s4BqKhHpk4cqoUCTEiu3rYjUjvty6gvpQy2BEUXqBkxUxkrH5+5Hty2Bb4zY21mxk3uaPKaktYUv9ljY1H/wuPwNSB1CYWkimN7NFACLgDvSIz9+L7KhfeLuc2hYXAlOMMbF/VGPMnLjFfisiF2FnV/y385qrlOpMIjIMeBl79Aw38BR2cdzvYAcadsgYMxM7WNnevLeAkR3MawQucR5KdZmSmjLu/+JpXl71HhFj8FhuvrPft7hk9BnkpmQBsKl2Ex+WzuGzss8JRoIA5PhymFR4OIf0n0DAHdit9zbGUBWsjGVHbK7f1GIkMb8rhYGpgxiYWsSAwEC8Lt8ef95E6RVRJ4hmRiQaxHBb7tgIGNGT/3CknXoQcRkXvemCpqeJmAiVjTWU11dQXr+dLXXllDdsZXtjBdVN1TSEGwhF7NoOXrcLv8uNJz7jwYIUL6TYmYGxyfa/p4VbPPhcdsG5bCfboTC1PwWp/cnwpOsFSR/g9/gpTO/HGcOmUdkwhdfWvcWamm/YUreVdG8qmb4MsvxplNZvoLR+A+meTAamFFOYMgCPy4fb5cZjuXGLC5fLhYU9OkVPqPUSPS5F69iEnEKU0VF84gOt0fo9bsvlrBMhGAnGsrjiR/RJduHKqHAk7GRGJBKMsGtGpHi8lNSWUFJbwhvrX6OktiQ2FFeUfSehPwNSB8QeOb6c2Pe9deAivuDwrmiznZ1to4PZoUhol987yZYDbhHZ1xizwpnWYd9vEbkEuxbFkcaYDTvZttMhUinVg/0Vu27MLGCrM+094K5kNUipzlBev52HvnyOZ1e8SSgSxiUWZww/lssP+A4FqXmETZgvtn7JnJI5rKxaFVtv36x9OaJwEiOzR+7WuWMoEmJzfSkbnIBETbBlrD/P349BTnZEji83adcvOw1OOH22XgTOMsY07Gz5nsbOPuj+zMzoCX/0hDRW+C2+HoSjeTSRji9UkvEZejpjDDXBOrY2VLK1voKy+m2U1W1la8M2qoJV1IXqCEYaiZhQLOjgdblwxdV48HvBjwX4W20bLHHhsXykuFJI92aQ47OzHfoH8sn0ZpLqSU3o4NAV/3b6+9D9PC4PeSk5+F0+Th82jaZwIy+seoXSunVUNdWyqdZNmiedQen5VAcr+Sr4Bd9UL6MwZSADA4NJ92TaNWyCFm6XG7fLhRsLsSxcuGL1bbrqj0F8Id3269g0dyMT7O5pRpyMrbguGpFY0CKucCXSpv5B/DKdIX47O/r9D4fDfFOxnsXly1lcvpwvt37TJhgxMK0fB+btywF5+zEmbxgh00hpbSmldSU8vfIztjdub7PdNE8ahYFCClMLKQgU0D/QH4/laREMCEWCtJgQC1QkJq5UaIdzdlfEhHvVccMYUysizwG/EpHLsIvjnQZMar2siJyHXSDvGGPMqlbzioEi4FPslLYfAXnAnNbbUUr1KBOBU40xERH7j5ExpkJEspLbLKV2T0VjNY8teZGnvn6NhnATgnDSkClcOfa7FKcXUhus5d0N7/HRpo+paKwAwGt5Gd9vPJMLD6d/oP8uv2d1UxUb6zawsXY9m+pKCJvmumA+y8eAaHZE6iD8Lv8OttR9dhqcMMaERGQ8dhGZXiVswjRFmjCEd77wbmgOQBC7i9icBt0qANFB7Ql7Qx0834s1hBrZ2lDJtoZKyusqKG/Yztb6rVQ1VVEbqqUx3EDYhHC7BJ/L3aabhdcDXo/QOugA2LUdLA8+y0+qJ5UsXya5/hzyU3LJ8GYknIIdMqGk/XsFI01tglyq64kI6b40PG4PFfWVTN/vu9QEa3nmm+fZ2riZ7ZHtlNdvw+dKZXTeUEKmifV1a1lft5Ysbw4DA8X09xcSNm6CYcsewUIsPC43VsSuVSNitcisiP89bO/isvVFe7QbRnQEjLAJE4qE7eyHiL1GrOCvREfyiWZOdBREddZx2hIflItEmpdr+Y3ZvYv0DrfRwQV6OBJmZeUGFpet4IvyFXy5dSX1oZZx9FgwIndf9skqpCFSS0ltKZvqVrBo+Rz7uxzHLW4KAgUUphZQmDqAAYFC0r0dlDrQ++9d6fvAw8AW7DunVxtjlojIFOA1Y0x0+JHbgVzg07jvyxPGmKuwa1Tci12NvwFYCJxkjNmKUqonqwKygPLoBBEZAGxOVoOU2h01TXX866tXeGLZf6kJ2uVSjimayNVjz2Z4VjEba0r4z4qn+bx8YSzLMc+fy6TCSUzoN54Ud0rC7xWOhNlcv4mNtespqVtPZVNli/m5vrxY7Yhcf17Ss3fbk2i3jn8APwTu7LqmdA17xI3dSy9unf1AByfu9h1HE1cHwoVoLYg2GkKNbG+sZntDFRWNVWxvqGZrw3YqGyupCtZQH6yjIdJAKBLEskws6OBzuWPZDm4PZHqgvaBDNNvBa/kIuFPJ8KaT488mPyWXbF9WrO+319W5VWXV3sfv8pEfyKWyoYqwK8zloy+mtG4Tz618kZpQBSFTx6ebFkPEx6SBBxKinoqmbVQ0bWO5tZQBKYMYmFpMmiedsLEIRoJ2MV3LhUfcdoFdE8buUuFcbDnX6cZEu4DZNU3sDIhwrMBudOH4wIaI2KML4xzHnG3Ej5YR/SlOIV+XU3x1p5kc3XRxHoqEKanZwrrqTfajqpR11ZvYUL2ZJqcfZlQ0GDE6bxgFqZnUhqoprS3lq6oFLNhW22bb2b5sBqQWUhgYwIDUQvJS8vr6SBi9gjFmG3B6O9NnYxfMjL4euoNtLAEO7Ir2KaW61HPAwyLyfQARycW+DtEhglWv0BBq5D/L/8ejS16gotHuQnFY4Vh+MPYcRuYO5cutS/jbF6+yump1bJ0RWfsxuXAyI7L3SzhwUBusYWOtnR1RWrexxQ0Xj+VlQGCgE5AYRMpu1qjoToleQR8MXCMiP8CuXhu7MjfGfKsrGranVlduZG3VRlLdKQxM7xcbgqW1WAAi9jOx7Idocbu9mTGG2mA92xurqWisoqKxmq31lWxv3E5FY5UzekU9TeFGQiaEyzJ4XS48Lhc+lwuP1bwP7UwHyMQHtFd0RXCLlxR3CmmeNLK8meSl5JLjzyY9ruJ9T4wAqr7JZbnICWTja/JR0VhJv5R+/Gjs1XxTsZL/rn4FqAVXiNkb5xIMezlpyBFg1VMVrGRt7SrW1q4ix5fHwEAx+f7+uC03EoEGE3aGBxY8Lg9usQhj16UJm4hTZdeuA2GP4uMU07Xsiih2FpcdrAg7w3q2qWUj9jGsJxbUbQw3saF6sxOA2MS66lLWV29iY80W+/O3ozA1nwPz92VE9iCy/QGqg5WU1pWycOsczNa2RSsLAwUUphYyIHUABYGCXboroZRSqlvcDDwIrHNebwH+hd2FS6keKxgO8sLKd3jgi2cpr7e7iR6UP5IfHHQuI3KKmbtpHs/P/08sq8Hn8nFIvwlMKjyc/JT8nW4/YiKU1W+O1Y6oaGrZFTXbmxMLRuSn9O9110aJnpl+4Dx6jVdXf8BDXz4Xe53mCZCfkm0/AtnkpWSR688kP2BPy0vJjg2BFw1C9MQT964WMRGqmmqpaKimorGabQ2VbGusZHvDdqqaqqkJ1tIQricYabK7VVj2cJlel4XX5cYdrecgTj0HbwddK4guZlf697vsbId0TxrZ/ixyfNmke9NJ96SR7k3Ha3n3+mCQ6plSvQE8Lg/bGyppCDYyPGsY14z9IV9uW8ob694AD6R4DP9b9zYmEuCM4cfjdgUprd/ItsZytjWW47N8DEwtZkBKEX53Cm7ciLgIhptoIpr5IERHFcbQnP0QaZnF5TyJ1bDpqd+b2mA96+OyIKLPN9VubbdOhSAUBHIpzihgQFoueSkZpHl9uC1he+M2SutKWVKxqc06/VL6tciKyPZ17RBYSiml9pwzXOh5IvJjYCiw1hhTluRmKdWhcCTMq6tnc9/i/1BSa/+qjswZyg8OOpfi9DzmbPqYZ1b+M5bZkJ+Sz+TCSYzPH4/fvePRMOpDdbHsiJK6jQQjTbF5bnFTGMuOKCLVk9p1H7IbJHT1bYy5rasb0tkKAtkcVzyOrQ0VVDXV2kPg0cCm+lI21Ze2u066J0CmP41MbzpZvnQyfWlk+tLJ8qWR6U0jzdu7xp+PmAgNoUbqQo3UhxqoCzo/Q/XUBuupC9XRGG6kMdRIk2lyirlFcFkWHsuF27JwWxIbTjXgg4DPzc6GmRfsoRO9Li9+l59Udwqp3jTS3AH8bj8+lw+/y4/P8mFZbfdnyxIcEaqClW2W6c0647KoIdhAhi+9w4wg1b28Lg/5KTmxAJ7f5efAvDGMzhnF/M0LeK/kfdK8ACGe+eZ5vJLFOSOm4XNHWF+7htpQDauqV7CqegX5/gIGpRaT481zulhYXZ7F1TxiR6RF17WICccyMZq7gpjoGMlxQYT4wY4h+h8D1AXr2dpQQXl9BVvrK9naUMG2hkpqgnX2J5HmoEpewE9BWjE5/nQyfAFSPT68LjeW2AUdmyKNzlBatVSHa6mpJ/aFyvSl4nVlk+5JJ82TRronjYAngIUVa9vmho1sbtjodG9p1WrTXD8o9pliy8QtGfvs0TXjt+X813S0X0yrqTua11yMtN193aYdzW1u+w72dr6bOh2XS7urKKV6D6c+jNaIUT1WxER4e91c7l30FGuqNgIwNHMgVx34PfICAeaUfsQLq9cC9vnOqOyRTC6czL5Zwzu8royYCFsbymLZEdsaW34FMr2ZsWBEP39BjxkhrTMknBogIkXAdOyq1+uBfyYwXFfSDM3OpZZ+QL892EoDjTSwubGMzY07X7pXsMDrtR/ZBIDu6HsUpiFSSUNTJTTtfGmVGJ/HS25KbrKboRyWZZHlz8RrealorEKM4HN5ObRwIgf3G8ec0o/4ZPMnZPlTgEYeWvI4md7+nDfiVDL8XtbXrGFTfQllDZsoa9iE35VCUepgAu40u85Ei6BBq9eEnYK87cyLe21aLd+69kSX7h835Ke7yU/Pwx4sYXckFoxriNTQ0FgDfeW43Wm04rJSSinVGYwxzCn5nL8t/DdfbbfrRgxK68+M0aeS5hM+2fQOVc5wnX6Xn0P6T2BSwSTyOjh3bwg3UOJkR2ys3UBTpPkkxiUuCgID7IBEYBDp3oyu/4BJklBwQkSOAF4HFgMrgXHAzSJyklOYqsdJ92bQz9+fpnCoxdCRHTEYguEQwUiQpnCIJuen/TpIUyRoV7zfCUssvJYbr8uDx/LgdbnxWh48Lvu5hRWrnh8yYcKRiPOzuap+8xB/4VjhTUuIFaoTEaeqv2BJtOL+rnEGLsUS++EWN27Ljcdy47JcuMR5OCMGqJ4lHAnjd2nWRE8U8KbgcbnZ1lBJfbAev9uPx+Xh6EFHcWjBRN7b8D4Lty4kPzUVY6r5y6L7KEgp5vxRpzIy6wA21q1jfc0a6sN1rKj6qtvaHesGEndcsJyRQ6LPQQhHIjSEG6kPxT2CDYSiQ5I6xydLBJeIPRqJZQ/hG80uiIrPAIi+tkTwWl68Lm/cTx8+lxevy4fX8hDLtZBoy+NKesamxc1rXrh5jrRYC6T5SGc/i9/WjrePtJwS3X786+Y2Ny/Z/vKt5rW3bWd6e3NaLt88ry7YgFs8KKWUUmrPzN+0hHsWPcmisq8ByE/J5tyR38LjDjKv7L3YkJ39U/oxuXAyB/cbh8/VsuuGMYZtjVvZWLueDbXrKW/Y0mJ+miedQU52RP+UQtzW3lFuINFP+Qfgx8aYh6MTRORi4A7gsEQ2ICI/BGYABwBPGmNmxM07DrgHKAbmAjOMMWsTbFu79s0cQUHKQLbWbSfg7ZxxWxvDTZTXV1BWt50t9dsoq9tOef12ttRtp6x+O2X126gPtX+rzhLBbVl4LAuPy2XXabBcsQKR3lYPdwIBlZbbd+GzfPjdflLdqaR708jwZpDqTiXgCRBwBwi4Uwi4A6S4U3pV9xTVVm1T7W6Nd9xbdPfxorN5XB76BXKpaqyhOliD1+XBbblJcadw0pCpTB4wibfWv8PXFcsoTE8nYrbxu/l/YZ+M/Zg+8tscWTCcrY1llNRtIGxCWLjaDRhIOwEE+3Xz8h0FG0KRMDVN9dQE66lqqqMmWEtlQy3VTbVUNlXZPxtrnNf2z2AkiNflwu92t3ikuD2kuD14XS52FitNcaeQ4ckgw2s/Mr3NzzO8Gfhdfq0J0QW0iLNSSim1Z74sX8E9C59k7qYvAMj2pXP68MlEpJYlFZ8C9i2B0Tn7M7lwMsMzh7X429sUbqSkbmMsO6IhXB+bZ4lF/5RCBqYWMSi1iHRPxl75dzvR4MQo4NFW0x4H/rQL71WCPRb5icTl5opIHvZwQZcB/wVmAU+RYNCjqxljCJmQXZsh3IhImEy/jxRPDgWpqTSG82mMNNIYbqIx1EhdyD7Jrw810BhpJBQJYZzq+7tCEPzuFFLdAdI8qQTiggypbifY4LxOcafgsfSOmOpTeuXxIp6IkOlPx+f2sL2hinCkCZ/bHsY2w5vBGfucRnnDZN5c9xZra9ZQlJlJfaSU2+b+if2zx3DuyJM4MOfgnb6PMYaGcCOVjdHAQnW7gYWqplqqGmuc41M9ESJ4LAu35bJ/upzgabTejMvC43aR77UotNLxuLIS+txpnrS2QQdPBpm+TNI96TqUr1JKKaV6leXb1/L3RU/x3gY7AJHtS+OEIQfRYCpZV7ccsG++TOx3CJMKDyfHnwPY52jbG7eywemuUVa/uUXR74A7NZYdURAYoNdzJB6c2Iw9nOj8uGkHYw/rkxBjzHMAIjIBGBQ360xgiTHmaWf+TKBcREYaY/Yop9kYQzASpCbYHFxofjTRGG6kKdzoBBeap0UfTc7rPe6PLfZdq2ghyOZMhriHJy7o4A7gc/n2ymiZUpCc40VX8bv95Ke4qWyspi5Yh9/lx7LsUTTyU/I5d79zKKkp4Y31b7KpvpQhWVlsD67hpjl/YHz+wQzNGtROsKGGmqZaGsINNIQbEUyb4ILHFRd4sCzS/C6yAh4syQayd+uz+Fw+UlwpZHjTyfBmtglCpHvS+1RRJqWUUkrtvdZVlfL3xU/xvzUfYTDkp6Rz2MB9aYhUsbWpBICCQAFHFE5iXP44vC4vwUgT62rWONkR66kL1cW2J4iTHTGIgalFZHl1BLHWEg1O3AW8KiL3AauBIcCVQGeM4jEaWBR9YYypFZGVzvQWFxsicgVwBUBxcfEON/rqmtd4b+P77Q5Jt6vcYo884XP54h5efJYPb+tp8a8tX2y9vaWfkFJdLOHjBezaMaMruV1uclKyqA3axTK9eGLHBBFhYPpAZoy6iNWVq3hj/VtsZxvDcrIpaVjOinVfOoEHJ+jgtsjzWuThZ0fD9HbEJW5S3H78Lj8pbj8+l/3c70xr+dwXe+5z+bQ7mFJKKaX6vNLaMh5Y/Az/XfUeBsPA9AxG5w8gZBqpC1cgCGNyx3BE4WSGpg+hKljJisqv2Fi3gS11m1rc2E5xpcRG1igMDNQM0p1IdCjRe0WkArsP+FnYo3X8xBjzZCe0IQ1oPW5xJe2MV2mMuR+4H2DChAk7jDq4xIXB4BY3PpevZXDB8sYFFezp0YJr0WW8seneTgksRAtbKtUZ9uKa+wkfL2DXjhldTURI86bicXmoqK+kMdKI1+VtLoQowj5Zw7g8Ywhfbf+Ktze8A9QQ8LRN8RPsopF+t58Up5aMHVTwtRNgsJ/7LF/CgdKOgrp6HLP1hiLB8UOQNg+JqpRSSqmOlNdv5+Evn+fZFW9iiWFoVhZDsnIwRAiZRgLuAIf2n8gh/SfQGK5nY+16Fm2dR02oJrYNQcj394sFJHJ8uZodsQt2epYqIm7szIlrOykY0VoN0Ho8lAygek82emzRMRzW/zCqGqvxeXw7X0GpXsQte+0oKl1yvOhOPpeXvEAOVY3V1IXq7YyEuAK4LsvF6NzRjMgewTcV3xAyIVLiAgx+lw+P5aVF5cl2fxWMM0taLNTeJWqr8SQQrPYX6Om6/Pp7998gkeBA64BC83rOdCfIEBt8xNhzJK5l4ryKjkASHTklxelOpJRSSqmWKhureXzpSzz51Wv43cKI3Bz6p6aBgCHCgNQBHNJvPNn+DErrSvjfhpeJmOZRHH0uPwMDdleNwtSB+F2dMxjD3minwQljTEhEzgF+2EVtWAJcFH0hIqnAMGf6bvNYHjyuEC7LrV0qVJ/jstx7axS2S44X3c1luchOycLX5KOisRIr4sLrbpkh4bbcjMge0eKidlcCUnvp70dSGWMwGCKx4VFNXEBBYj9xnkeHXsU480QQsRBnOFDLWdayJG6oV4fED3Ea/fduPXxps7AJdeVHV0oppXqd2mA9/1r2Ck8se4k0n5sD++eR7rNvartwsX/uCAakFVATrGJVzVf2LTJHri+fgamDGJRaRI4/T7u+dpJEr9pfwu7O8czuvpGTgeEGXIBLRPxACHgeuENEzgJeAW4BFndGcbvo6VkwHIxvB2DfSRJET+BVj2JMy7urre+2Rl/39RTtZBwvkiHgTcHtclPRWEV9sAG/u2UhXBHZpYCE6nzRgEM0yBB9bWcnWBgiRP/a2MO1Cu7YMK/Nf2fsbIfmf0+hOYBkTHNAoaPvvPOieVq0PfZmiZhIm98VY0zz++nfOqWUSpqmcBNg3zzV43HyNYQaeWb5G/zjqxfJ8Lk4qLAfXpcLj+UmNyWbgWkFhE2QsAlSUrceAK/lZYBTyHJAYCAp7kCSP0XflGhwwgM8ISJXAWugucqHMeaKBLfxS+DWuNfnA7cZY2Y6Fxp3A08Ac4FzEtzmjhvt8pDuS8Ulln0qaSKETYRwJEKEiD3MJ81BDANYccELjYCpeIkGDqI/4i8WTPQywtivI0TnGWdZQ/Q+qBET+520LCsuk7z5TqnX6vOjuXT78SJZvC4P+Sl2N4+aYC1el1dHvOhirQMOxLIdmi/mETtoYCGIJbjEQsRl/4QW3WpEBIllP7R9r/iggYn9+ZQWxxT7b5HzDY81QWJTAbCal2n9/Y9vd5tp0a307WOGUkr1WPd+8TAbalfHgtEiFhYWluXCLW7clgeP5cFreZzadz5SnMLV9ih/flI9qaR5AvjdKXgtLz6XF6/Li9fy4XV59LolAcFwkBe+eYcnl79Ius/igH45pHkDpHsD5PizcDndH5siDQBk+3JitSPy/f10H3eDRIMTQSBab8LlPHaJMWYmMLODeW8BI3d1m4mwU2Gd5orL/sBxrbdPUCNEMGCME7wIEyZCMBxs7scbPVEVJ9lW0F/QHqrlCb9pMT165xOaAwKxwAHE+nJHIs3zotGB5n/v5juR4ryHJRYYia1j3z11x/p9R39f4u+bSuyuatw8ibuckNiSsc8Qn8Ldl3//knW8SBYRIdOfgdflZXtjJaFIGJ9bqznvKmPsIEP0e2t/1aPZRvZ/7XoaBsuynO+dk+HgPKKHjOh1vCVWLMBgBw6dLAk7UtAcLLCav5NtAwwtaTaDUkrtXWqCtaR6UprPO2PnpBEiNNEYbqQhbFqcq5oWZ7E71zLw4cISF27LjVvceFzeWEDD5/LZo3a5/QTcAVI9KaS6A6R5Ukn1pMYGDPA6D5f0/hsm4UiYl1e/z9MrXiQ7xcOEAQNJ9wZI8wRa3BDyWB4KAwPtgERgEAFPahJbvXdKtCDmMuCvxpj6rm9S97K/xC67/JvQTvAiYp/wOgeKsAkTiUQImQihSDAWtIie9Np9hZtPdvui1hkEO1y2dXZBB4EDIK6fdnzasp1hEJ0VIYIlzQd2i+aAQYQIVouTfvtiIVpAzsLCEhdixQcK4oIFVlx3HydQENueFXfREXtH4t5LLzjUnkvx+PFYbioaq2kINeBz9c0MmfjCj/Zr4l63nSaxn81FPWPTxDk+GBOr12BhISLOCYczneYMB7HaZhTEf+/jf0YDEPHLKqWUUrti0oCDWV+7crfWjZhIi+B7LAgf694X/bvZKrDRojtiBGPqaTB11AcN24Ntuyt2tH4kYpybuGDsfEI7wxwXLrGHRo8fFdEfy/iID3oESPemEfCk4LN8saBHV/9NNcbw31Vv8/7G9+mflsqUon1J8bQsVpnpzWJgahGDUovIT+nfJ4IxvVmiBTFvMsb8oTsa1NlE2hYCa75LLtEJbbQoQufczbbvVrvBAp+zTMREiDjBCWMMoUiYiAkTjEScE/DmfsmC1XxS3MH1vYkbFzeRvuatL/DbWyd+idaBhZbpxnHZAzQf8KJ3E6Mpxi3Wl7b7Mz49OVbkrfX7ScsLAgB3i/7YTnAgWgjOudiIHsOiGQlALEBhxQ3JGP9eejGhehO3y01uShbVTbVUN9U4f7zbLrfDGOEu/MrvXv0S5ySlVQChubtSc12F+ABCfDZSi5oMcQ2O1m2wv+ItM4miAYpYZwdpdbxwshnif7Y4HmigQSmlVBIUpQ3EEHSuG8KETcR+TtiZFrFvgEanR58TsbPyZDfS1ruZHdRoIGTqqQobKkPtBz/s4Ird3d5g35xsvmkhLf8nFi5x4RJ7gINo1xevyx69LNrtJdUdINUTwO9KsYdpN8JbG95jY+1aMv2pjC0YHGunIBQGBlKUVszA1CLSPO2ORq+SJNFuHe+KyFHGmPe7tDWdzB03Ukd7d/sTOSlvd5mdBDNi05wvYTgSaa53YezgQ4s78O2cILd3ztw68NBeH+OElm9vWWl9iUAsgND+tlqu29H76Mm/UrtORMjwpeFzeagN7jhhbWdBzMS/gTtZstVsV1xXJCsWNYy78HeWa9HVAeeYkMDxrf0mtHes1GOMUkqpnm145n4Mz9xvl9ezs7dbBizC8cGLaLAjbpl250entVmu5bIhEyIYCRGKf5iQ3eXdhJ1giom7mdp8YzM+47BrhIEwTTTQFK6mJgw07XiN3IA9+nwoHKE4fTCjskdTkFKAS0dy7LES/ZdZA7woIs/QtiDmbzq/WZ2v3QDA7n57dnW1nh7qVEr1SD63D5/bl+xmKKWUUioJRAQXLqergWenyydTczAkLpBC3PNImKZIE/WhBupDjdQF66kL1VEfbqAh1EBTpJGmcBPBSJBgJETYhJxsEnv95jGqosEQmofedm6URGuyCUJjuAnL+Jk25CQGZwzeYdtVz5FocOIg4HNgmPOIMkCvCE4opZRSqvcQkRzgIeBbQDnwC2PMv9pZToDfAZc5kx4EbjROyqSIHORsZxR2Da1LjTELu7r9SqneK9Hjj2pmd8tMXrH2iInQGG6iNlhPVVM1han9SHH7d76i6lESCk4YY47p6oYopZRSSsW5Bztptz/2TZJXRGSRMWZJq+WuAE4HxmLfNHkTWA38XUS8wIvAncDfgCuxM0H3NcbsJCFYKbUXS/T4o3oISyxS3HYxzryU7GQ3R+2mhMNbIuISkUkicrbzOiAiKV3XNKWUUkrtjUQkFTgLuNkYU2OM+RB4CbigncUvAv6fMWaDMWYj8P+AGc68o7FvxNxpjGk0xvwFOyP42C7+CEqpXmoXjz9KqU6UUHBCRIYBXwKvYqc4gZ3m9EAXtUsppZRSe6/9gJAxZnnctEXA6HaWHe3Ma2+50cBi07Iq9uIOtqOUUrBrxx+lVCdKtObEX4F/A7OArc6094C7uqBNCVmwYEG5iKxN1vvvgTzsvmuqY7qPdi7RffS6MWZqVzemN9BjRp+m+2jnEtlHPel4kQZUtZpWCbQ35luaMy9+uTSnFkXreTvaDiJyBXY3EYAaEfl6J+3U373uo/u6e/W2Y0ZnSuj4o8eLHk/3d/fptOuSRIMTE4FTjTERETEAxpgKEclKcP1OZ4zJT9Z77wkRmW+MmZDsdvRkuo92TvfRrtNjRt+l+2jneuE+qgEyWk3LAKoTWDYDqDHGGBHZle1gjLkfuD/RRvbC/dpr6b7uXnv5/k7ouKHHi55N93f36cx9nWjNiSogq1UjBgCbO6MRSimllFJxlgNuEdk3btpYoL1idEucee0ttwQ4UFqOJ35gB9tRSinYteOPUqoTJRqceA54WEQGAYhILnbl6393UbuUUkoptZcyxtRin3v8SkRSRWQycBrwj3YWfxz4mYgMdG6cXAs86sx7DwgDPxYRn4j80Jn+Tle2XynVe+3i8Ucp1YkSDU7cjJ3KtA47g2IL0Aj8pmua1aclnP61F9N9tHO6j/Ye+m+9c7qPdq437qPvAynY5xxPAlcbY5aIyBSnu0bUfcB/gS+wi3e/4kzDGS70dOBCoAK4BDi9E4cR7Y37tbfSfd299vb93e7xZw+3ubfv0+6m+7v7dNq+lpYFrHeysJ0xMRRYa4wp66xGKKWUUkoppZRSau+1S8EJpZRSSimllFJKqc6WaLcOpZRSSimllFJKqS6hwYlOJiI5IvK8iNSKyFoRmd7BcjNFJCgiNXGPfbq7vckgIj8Ukfki0igij+5k2Z+KyCYRqRKRh0XE103NTKpE95GIzBCRcKvfo6O7raGq04jIeyLSEPfv+HXcvOnO8aRWRF4QkZxktrW77Oh7ICLHichXIlInIu+KyOC4eT7neFHlHD9+1u2N7yYd7SMRGSIiptWx4ea4+XvNPlJKKaVU76DBic53D9AE9AfOA+4VkdEdLPuUMSYt7rGq21qZXCXA7cDDO1pIRE4EbgSOAwYD+wC3dXnreoaE9pHj41a/R+91bdNUF/ph3L/jCADn+HEfcAH2caUO+FsS29id2v0eiEgediX1m4EcYD7wVNwiM4F9sY8bxwA/F5Gp3dDeZNjZsSIr7ndqVtz0mew9+0gppZRSvYAGJzqRiKQCZwE3G2NqjDEfAi9hX1QohzHmOWPMC8DWnSx6EfCQMWaJMWY7MAuY0cXN6xF2YR+pvu884L/GmA+MMTXYF+Rnikh6ktvV5XbwPTgTWGKMedoY04B9oT1WREY68y8CZhljthtjlgEP0EePHXtwrNhr9lFnE5FhIrJNRA52Xg8QkTLNWut8InK9iDzbatpfROSuZLWpLxORs1tlWzWKyHvJbldvpseL7qXHjO7VFccMDU50rv2AkDFmedy0RUBHmROnOAesJSJyddc3r9cZjb3/ohYB/Z1RY1SzcSJSLiLLReRmEXEnu0Fqt/3W+becE3fi0uJ7YIxZiZ2dtV/3N6/HaL1PaoGVwGgRyQYKaXvs6Og43NetFZENIvKIk3GC7qM943wHbwCeEJEA8AjwmGatdYkngKkikgXg/H07B3g8mY3qq4wxsYxeYACwCnsYTbWb9HjR7fSY0Y264pihwYnOlQZUtZpWCbR3h/M/wCggH7gcuEVEzu3a5vU6adj7Lyr6vM/fMd4FHwBjgH7YWTvnAtcntUVqd92A3XVpIPZ40f8VkWG0/R5Ax8eVvcWO9kla3OvW8/Ym5cAh2N02xmN//n8683Qf7SFjzAPAN8Bc7EDP/yW3RX2TMaYU++/cd51JU4FyY8yC5LWq7xMRC/gX8J4x5r5kt6e30+NF99FjRnJ05jFDgxOdqwbIaDUtA6huvaAxZqkxpsQYEzbGfATcBXynG9rYm7Ten9Hnbfbn3soYs8oYs9oYEzHGfAH8Cv096pWMMXONMdXGmEZjzGPAHOBkduG4shfZ0T6piXvdet5ew+laON8YEzLGbAZ+CHzL6Q6k+6hzPIAdHP6rMaYx2Y3pwx4Dzneenw/8I4lt2Vv8GjtY+eNkN6QP0eNF99FjRvfrtGOGBic613LALSL7xk0bCyxJYF0DSJe0qvdagr3/osYCm40xWoehY/p71HdE/y1bfA/EHtXHh3282Vu13iepwDDsOhTbgVLaHjsSOQ73Zcb5aek+2nMikgbcCTwEzNxbRtBJkheAA0VkDDCN5gwg1QVE5BzsLMzvGGOCyW5PX6DHi273AnrM6DadfczQ4EQncvo9Pwf8SkRSRWQycBrtROxE5DQRyRbbROxI04vd2+LkEBG3iPgBF+ASEX8HdRIeBy4Vkf2dvmO/BB7tvpYmT6L7SEROEpH+zvOR2MUS94rfo75ERLJE5MTov7OInAccCbyO/Uf1FBGZ4lyE/wp4zhjT5+9y7+B78DwwRkTOcubfAiw2xnzlrPo48EvnGDsSu+vco0n4CF2uo30kIoeKyAgRsZw6PX/BTreMduXYa/ZRF7kLmG+MuQx4Bfh7ktvTZzlFb5/BThmeZ4xZl+Qm9VkiMg74K3C6MaYs2e3pQ/R40Y30mNF9uuSYYYzRRyc+sIe1ewGoBdYB053pU4CauOWexK6uXgN8Bfw42W3vxn00E/suXvxjJlDs7I/iuGV/BmzGruXxCOBLdvt70j4C/ujsn1rsIjS/AjzJbr8+dvnfOx/4FDutvgL4BDghbv5053hSix18ykl2m7tpv7T7PXDmHe8cO+uB94Ahcev5sIfWrHK+Hz9L9mfp7n2EfRdjtfM7U4odjCjYG/dRF+zz04CN0e8hdg2Pb4Dzkt22vvoAjnB+ty9Odlv68sM5doSc84zo47Vkt6s3P/R4kbT9rseM7tnPnX7MEGfDSimllFJK9TgiUowdjCwwxrQuPK6UUi3oMaP30m4dSimllFKqR3KqwP8M+LdeZCildkaPGb1be/38lVJKKaWUSiqnzs5mYC32kIBKKdUhPWb0ftqtQymllFJKKaWUUkml3TqUUkoppZRSSimVVBqcUEoppZRSSimlVFJpcEIppZRSSimllFJJpcEJpZRSSimllFJKJZUGJ5RSSimllFJKKZVUGpxQSimllFJKKaVUUmlwQimllFJKKaWUUkmlwQmllFJKKaWUUkollQYnlFJKKaWUUkoplVQanFBKKaWUUkoppVRSaXBCKaWUUkoppZRSSaXBCaWUUkoppZRSSiWVBieUUkoppZRSSimVVBqcUEoppZRSSimlVFJpcEIppZRSSimllFJJpcEJpZRSSvU4IvJDEZkvIo0i8uhOlv2piGwSkSoReVhEfHHzhojIuyJSJyJficjxXd54pZRSSu2ynQYnRORoEblTRD4QkS+dn3eJyDHd0UCllFJK7ZVKgNuBh3e0kIicCNwIHAcMBvYBbotb5EngcyAX+D/gGRHJ74oGK6WUUmr3iTGm/Rl28OFOIBt4G/gCqAIygDHYJwEVwE+MMe92Q1uVUkoptZcRkduBQcaYGR3M/xewxhhzk/P6OOCfxpgCEdkP+/wlzxhT7cyf7cz/e7d8AKWUUkolxL2Deb8GrgfeNO1EMEREgBOAWcARXdO8jk2dOtW8/vrr3f22SvU2kuwG9BR6zFBqp3rr8WI08GLc60VAfxHJdeatigYm4uaPbm9DInIFcAXA/vvvP37JkiXtvuGSxZuYN3dtJzRdJcoYQ3Z2gFPOGIPbrb2Se4jeeszodHqOoVRCdnrM6DA4YYyZtKMVnYDFG86j25WXlyfjbZVSvZQeM5Tqs9KAyrjX0efp7cyLzh/Y3oaMMfcD9wNMmDCh/dRSoF9BKvn5adF1wIDBYAzOc8C5r2Pi5zkT7Pk0T3PmG0zc+rE5rZZ13pP2p8XWx7TaDi03HL9+3Dqtt9tm+22mt5myg3m7xhhA7LNZt8eioqKeRx+cywknjmDw0Jw927hSnUjPMZTqHDvKnFBKKaWU6ulqsLucRkWfV7czLzq/mj2Q3y+daae3m3yhusjcj9ayeFEJbrfFm//7mgEDMpk6bRSWpTfvlVKqr0goL05EUkXkFyLyrIi8Ef/o6gYqpZRSSu3AEmBs3OuxwGZjzFZn3j4ikt5qfvv9NVSPdeikwXznnLFYIrhcFps3V/PwfZ+wedMexZmUUkr1IIl22nscOB/4BpjT6qGUUkop1alExC0ifsAFuETELyLtZXw+DlwqIvuLSBbwS+BRAGPMcmAhcKuz/hnAgcCz3fARVCfLzg5w4aWHMHhIDpGIweW2ePHZL3j3zRUddjtRSinVeyTareM4YIgxpqIL26KUUkopFfVL4Na41+cDt4nIw8BSYH9jzDpjzOsi8gfgXSAFO/AQv9452MGK7cA64DvGmLJuaL/qAiLCsSfsS8mGfvzvta/weF2sXr2VVX8v58yzx5KdE0h2E5VSSu2mRDMn1hNf4UgppZRSqgsZY2YaY6TVY6YTkEgzxqyLW/ZPxpj+xpgMY8zFxpjGuHlrjDFHG2NSjDEjjDFvJecTqc40YFAm58+YQH5+GiKC5bZ48h+f8fHs1ZpFoZRSvVSimRM/Ae5z7kxsip9hjCnp7EbtiUgkQnl5ORUVFYTD4WQ3R6ku43K5yMrKIi8vD8vSYdWUUkrtXTweF9NOH83yr7cw5/3VpAQ8fPnlJr5asoXvTB9LeoY/2U1USim1CxINThhgCvDduGniTHd1dqP2xIYNGxARhgwZgsfjQUSrOKu+xxhDMBhk8+bNbNiwgeLi4mQ3SSmllEqK/Ub0Y9CgLF55aSk1NXbSzCP3zeWIo/fh4EOKktw6pZTquzasr+C1F5cxfcbBpKb59nh7id5uvQ+7v+YYYB/nMdT52aPU1tYycOBAvF6vBiZUnyUieL1eBg4cSG1tbbKbo5RSSiVVINXLd84ZywFjCzHGkJ7lZ94n6/jHg59SW9O48w0opZTaJdu21vHw3+ay4qsy3n9rZadsM9HgRH/gl8aYZcaYtfGPTmlFJ9MUd7W30N91pZRSyiYiTJhYzKlnjMHrdeHzu2kKh7nvrx/xxcIe1QtZKaV6tbraJh762yfUVDcyfL88vjVtZKdsN9Erm7eA8Z3xhiKyr4g0iMgTcdOmi8haEakVkRdExU2HggAAZnFJREFUJKcz3ksp1fvpMUMppdSuyMtP43vTxzF4SDaWJWTnBXj37W948tEF1NU1Jbt5SinVqwWDYR69bx5lm2soGJDOBZcfgtvdOTdME93KauAVEfmriNwU/9iN97wH+DT6QkRGY3cbuQA7Q6MO+NtubHevMnv2bLKysna4zPDhw3n00Ud3uMw555zDQw891HkN6wLr1q0jLS2NkpKeedcjfj8vWbKEESNG0NioKaSdSI8ZSimldonH4+LYE/bj6GOHY1lCeoaPmrpG7vl/H/LVks3Jbp5SSvVKkYjh3499xppV28jM8nPJ1YeRkuLptO0nGpw4GHtM8THACXGP43flzUTkHKACeDtu8nnAf40xHxhjaoCbgTNFJH1Xtt1bLViwgLPOOot+/fqRlpbGkCFDOOuss3jnnXd2uN6UKVOoqKjYo/f+5JNPmDdvHjNmzNij7XS14uJiampqGDBgQJe/1w033MDo0aPJyMhgwIABXH755Wzbti3h9UePHs3BBx/M3Xff3YWt3HvoMUMppdSeGDosl7POHktObgC3x0VeQSovv7CEp//5OQ31wWQ3T3UhEXlCREpFpEpElovIZXHzjhORr0SkTkTeFZHBcfN8IvKws94mEflZcj6BUj3Py88v4YuFpfj9bi65+jCyslM6dfsJBSeMMcd08Dg20TcSkQzgV0DrL/hoYFHce60EmoD9Et12b/Xmm28yefJkhg0bxvz586muruaLL75g+vTpPP/88x2uFwx2zh/Tu+66i4svvhiXq0cNuJJULpeLJ554gq1bt7Jo0SI2bNiwy8GbSy65hL/+9a9EIpGuaeReQo8ZSimlOkNamo9TTh/DuPGDAMjtl0pZeS13/eEDvvm6LMmtU13ot8AQY0wGcCpwu4iMF5E84Dnsmxs5wHzgqbj1ZgL7AoOBY4Cfi8jU7my4Uj3RB++s5MN3V+FyCRdefgiFAzM6/T26s5reLOAhY8yGVtPTgMr/3959x0lVnQ0c/z0zs73Se5euIEXEgg0RRFQUY8Vg1NhiovFNYkxMJJoYExMTNcYWsccSe8GOKKiAoIA06VVA2sL23Zl53j/und2Z2TYLuztbnu/nM+zcc8899+zdncPe554SlbYfqPAUVESuEpGFIrJw166m/5/Jtddey9SpU/nrX/9K9+7dEREyMjKYMmUK999/f1m+k046iRtvvJHJkyeTmZnJ3//+d2bPno3PV74SbGlpKTfddBPt27enY8eO/OUvf6n23H6/n7fffptx48ZFpM+fP58RI0aQkZHB8ccfz+23307Pnj3L9t97770MGDCAjIwMunfvzi233EIgECjbLyLMnTu3bDu6ns8//zwDBw4kIyODDh06MG3aNMBZGvO3v/0tnTt3JiMjg549e5Zdg40bNyIibN3q/OosWbKEE088kbZt29KqVStOP/101q0rnyH2sssu49JLL+XHP/4x2dnZdOnShYcffrjGnwfAnXfeybBhw0hISKBdu3bccMMNzJ49u1bX+YQTTmDHjh0sXrw4pnOaKlmbYYwxpk54PMKRw7twxlmDSElNICU1gbYd0njh2a959YUllBT7411FU8dUdbmqhsbZqvvqA5wLLFfV/6lqEU4wYqiIhGb0mwbcoar7VHUl8ChwWYNW3phGZslX23jrleUAnD91GIf1b1cv5/HVnAVEpBTnA12BqibGcPyROENAhlWyOw+IDrtkArmVnOsR4BGAkSNHVlqfaL+6/o1YstWZv/7rrJjyrV69mnXr1sV80zxjxgxee+01Xn31VQoLC1mwYEHE/rvuuou33nqLzz//nC5dunDTTTexaVPVi6msWbOG3NxcBg0aVJaWk5PDxIkT+fWvf82NN97IsmXLmDRpEgkJ5eOIunbtyjvvvEPPnj1ZvHgxEyZMoGfPnlx99dU1fg8FBQVceumlvPfee5xyyink5+fz1VdfAU4vkieffJL58+fTrVs3vv/+e7Zt21ZpOSLC9OnTOfbYYykqKuLKK69k6tSpfPHFF2V5XnrpJV544QUefvhhXnvtNS644AImTJhAjx49Ki2zKh999BFDhw4t247lOiclJdG3b1+++uorhg8fXqvzGUc82wxjjDHNV/sOGZz7gyF8PmcDG9bvpXO3LDZvzuGfd83m/KnD6NmnTbyraOqQiPwbJ7CQAnwNzAT+RGQPzHwRWQcMFpGdQKfw/e77yQ1UZWManfVr9/D8U18DcPrZAxl2VNd6O1esPSdOJXKuictwPqg3xnj8SUBPYLOI7AB+AUwRka+A5UDZ3Z+I9AaSgNUxlt0khZ7idunSpSztjTfeIDs7m6ysLJKTkyPyn3feeZxyyimICKmpqRXKe+qpp7j55ps57LDDSElJ4W9/+xsiUuX59+3bB0BGRvnD5rfeeov09HR+8YtfkJCQwLBhw7j88ssjjpsy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WFjIggULIvbfddddvPXWW3z++ed06dKFm266iU2bNlVZ3po1a8jNzWXQoEFlaUVFRZxyyilcfPHFPP300xQVFXHJJZdwww03MGPGDK644go+/fRTLrnkEmbOnMnFF1/MuHHjuOKKK8rK+PTTTxk3bhzbt2/nm2++4fTTT6d79+5cfPHFbNiwgQkTJvDggw8ydepUFi5cyMSJE2ndujU/+MEPAAgEArz//vssWbKE/Px8jj/+eO677z5++9vfoqpMnjyZ/v37s2zZMhISEvjpT3/KxRdfHDEvw4wZM3j99dcZMGAAv/jFL5g2bRpr1qzhggsuYOXKlcydO5cPP/ywLP/mzZsjrk1SUhL/+te/GDZsGLt37+b888/nhhtu4LnnnovpZxUtOzub/Px8/H4/J5xwAqeddhoA69at49xzz+Wxxx7jwgsvZNasWZxzzjkcddRREccfccQR1fakMcaYRuR7Ebka+AbwisgxVDK/g6p+3uA1M6aBiQiH9WtH+w4ZfPLxWnbvyqdrr2xKiwOsXbWbGQ/O56hjunPmuYNrtYxfcyYi1+AEJdbgrLBzDTAK+B2wQESuUdWG/ePemCaqqLCUGQ/OI2dfIT16teKiaSPweGoOTBQVlfLBO9+ye3c+qWkJjD99INmtYlvR41DF2nPiCWA+zljPJ920UcDaqg4w1QstUdalS/l4nzfeeIPs7GyysrJITk6OyH/eeedxyimnICKkpqZWKO+pp57i5ptv5rDDDiMlJYW//e1v1UbF9u3bB0BGRnkE7K233kJVuf3220lJSaFVq1bccccdPPvsswQCAQAefPBBvvvuO0aNGsWOHTt48MEHI8rt1KkTN998M4mJiYwYMYKrrrqqbFLO5557juHDh3PZZZfh8/kYPXo0V199Nf/5z38iyrjrrrtIT0+nQ4cOTJ48mYULnYdcixYtYtGiRTzwwANkZWWRmprKX//6V2bNmsXWrVvLjr/66qsZPHgwXq+XK6+8krVr17J///4qr0W0448/nqOOOgqfz0fHjh351a9+dUiTUubk5JCXl8err77KxIkTy3ppPP/884waNYqpU6fi8/k47bTTmDx5coXjMzMz2bt370Gf3xhjGtBPgZtwJq1LxpnbYW7Ua07camdMHGRmJTPxzEEcMbQTAAlJXo4c1YWkFB9ffrGZe+6czZpvbela123AcFU9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\n",
      "text/plain": [
       "<Figure size 1080x504 with 12 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot_comparison_precision_2([benchmarks_ms, benchmarks_radius], colors=colors)\n",
    "plt.tight_layout()\n",
    "plt.savefig('influence.pdf')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "benchmarks_cutoff['Monopolar (peak_voltage)'] = benchmarks_cutoff.pop('Mononopolar (peak_voltage)')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "benchmarks_cutoff.keys()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "tmp[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from spikeinterface.sortingcomponents.benchmark.benchmark_tools import BenchmarkBase, _simpleaxis \n",
    "import pylab as plt\n",
    "import matplotlib\n",
    "\n",
    "def plot_comparison_precision_2(benchmarks, colors=None):\n",
    "\n",
    "    import pylab as plt\n",
    "    fig, axes = plt.subplots(nrows=3, ncols=len(benchmarks) + 1, figsize=(15, 7), squeeze=False)\n",
    "    \n",
    "    to_explore = list(benchmarks_ms.keys())\n",
    "    to_explore.remove('xaxis')\n",
    "    \n",
    "    for title in to_explore:\n",
    "        \n",
    "        \n",
    "        if title.find('Monopolar') > -1:\n",
    "            jcount = 1\n",
    "        elif title.find('CoM') > -1:\n",
    "            jcount = 0\n",
    "        elif title.find('Grid') > -1:\n",
    "            jcount = 2\n",
    "    \n",
    "        for icount, benchmark in enumerate(benchmarks):\n",
    "\n",
    "            bench = benchmark[title]\n",
    "            \n",
    "            #vrange = np.array(list(bench.keys()))\n",
    "            #v_min = np.min(vrange)\n",
    "            #v_max = np.max(vrange)\n",
    "\n",
    "            #my_cmap = plt.get_cmap(cmaps[jcount])\n",
    "            #cNorm  = matplotlib.colors.Normalize(vmin=v_min, vmax=v_max)\n",
    "            #scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=my_cmap)\n",
    "\n",
    "            if icount == len(benchmarks) - 1:\n",
    "                b = bench[0]\n",
    "\n",
    "                spikes = b.spike_positions[0]\n",
    "                units = b.waveforms.sorting.unit_ids\n",
    "                all_x = np.concatenate([spikes[unit_id]['x'] for unit_id in units])\n",
    "                all_y = np.concatenate([spikes[unit_id]['y'] for unit_id in units])\n",
    "                try:\n",
    "                    all_z = np.concatenate([spikes[unit_id]['z'] for unit_id in units])\n",
    "                except Exception:\n",
    "                    all_z = np.nan * np.zeros(len(all_x))\n",
    "\n",
    "                gt_positions = b.gt_positions\n",
    "                real_x = np.concatenate([gt_positions[c, 0]*np.ones(len(spikes[i]['x'])) for c, i in enumerate(units)])\n",
    "                real_y = np.concatenate([gt_positions[c, 1]*np.ones(len(spikes[i]['y'])) for c, i in enumerate(units)])\n",
    "                try:\n",
    "                    real_z = np.concatenate([gt_positions[c, 2]*np.ones(len(spikes[i]['z'])) for c, i in enumerate(units)])\n",
    "                except Exception:\n",
    "                    real_z = np.nan * np.zeros(len(real_x))\n",
    "\n",
    "                dx = np.corrcoef(np.nan_to_num(all_x), real_x)[0, 1]\n",
    "                dy = np.corrcoef(np.nan_to_num(all_y), real_y)[0, 1]\n",
    "                dz = np.corrcoef(np.nan_to_num(all_z), real_z)[0, 1]\n",
    "                #dx = np.nanmean(all_x / real_x)\n",
    "                #dy = np.nanmean(all_y / real_y)\n",
    "                #try:\n",
    "                #    dz = np.nanmean(all_z / real_z)\n",
    "                #except Exception:\n",
    "                #    dz = 0\n",
    "                \n",
    "                \n",
    "                #ax = axes[jcount, icount+1]\n",
    "                \n",
    "                #x_means = np.array([np.nanmean(dx), np.nanmean(dy), np.nanmean(dz)])\n",
    "                #y_means = np.array([np.nanstd(dx), np.nanstd(dy), np.nanstd(dz)])\n",
    "                x_means = np.array([dx, dy, dz])\n",
    "                \n",
    "                ax.plot(np.arange(3), x_means, c=colors[title], lw=2)\n",
    "                #ax.fill_between(np.arange(len(x_means)), x_means-y_means,x_means+y_means,\n",
    "                #            color=colors[title], alpha=0.05)\n",
    "                _simpleaxis(ax)\n",
    "                \n",
    "                ax.set_ylabel('corrcoef')\n",
    "                #if jcount == 0:\n",
    "                ax.set_xticks(np.arange(3), ['x', 'y', 'z'])\n",
    "                ax.set_ylim(0, 1)\n",
    "               # ax.set_ylim(0, 45)\n",
    "            \n",
    "            ax = axes[jcount, icount]\n",
    "            \n",
    "            _simpleaxis(ax)\n",
    "\n",
    "            x_means = []\n",
    "            y_means = []\n",
    "            y_stds = []\n",
    "            labels = []\n",
    "            \n",
    "            for b in bench:\n",
    "                x_means += [np.nanmean(b.medians_over_templates)]\n",
    "                #x_stds += [np.std(b.medians_over_templates)]\n",
    "                y_means += [np.nanmean(b.mads_over_templates)]\n",
    "                #y_stds += [np.std(b.mads_over_templates)]\n",
    "                #colors += [scalarMap.to_rgba(key)]\n",
    "                #label = b.title.replace('Mononopolar', '')\n",
    "                #label = label.replace('CoM (ptp)', '')\n",
    "                #label = label.replace('Grid', '')\n",
    "                #label = label.replace('[', '')\n",
    "                #label = label.replace(']', '')\n",
    "                #labels += [label]\n",
    "                #title = b.title\n",
    "            xaxis = benchmark['xaxis']\n",
    "                #ax.scatter(x_means, y_means, c=colors, label=label, s=200, edgecolor='k')\n",
    "            \n",
    "            x_means = np.array(x_means)\n",
    "            y_means = np.array(y_means)\n",
    "            ax.plot(xaxis, x_means, color=colors[title], lw=2, label=title)\n",
    "            ax.fill_between(xaxis, x_means-y_means,x_means+y_means,\n",
    "                            color=colors[title], alpha=0.05)\n",
    "                \n",
    "            #ax.errorbar(x_means, y_means, xerr=x_stds, yerr=y_stds, fmt='.', c='0.5', alpha=0.5)\n",
    "                \n",
    "    \n",
    "            #ax.legend(loc='lower right')\n",
    "            \n",
    "            if icount == 0:\n",
    "                ax.set_ylabel('error medians (um)')\n",
    "            else:\n",
    "                pass\n",
    "                #ax.set_yticks([])\n",
    "            \n",
    "            if jcount == 2:\n",
    "                if icount == 0:\n",
    "                    ax.set_xlabel('Time (ms)')\n",
    "                elif icount == 1:\n",
    "                    ax.set_xlabel('Radius (um)')\n",
    "                elif icount == 2:\n",
    "                    ax.set_xlabel('Cutoff (Hz)')\n",
    "            #else:\n",
    "            #    pass\n",
    "                #ax.set_xticks([])\n",
    "                #ax.set_xlim(7, 9)z\n",
    "            #    ax.set_xticks([])\n",
    "            #else:\n",
    "            #    ax.set_xticks(np.arange(len(labels)), labels, rotation=45)\n",
    "                #ax.set_xlim(12, 14)\n",
    "            \n",
    "            #ymin, ymax = ax.get_ylim()\n",
    "\n",
    "            if icount < 3:\n",
    "               ax.set_ylim(5, 40)\n",
    "            elif icount == 3:\n",
    "                ax.set_ylim(0, 1)\n",
    "            #else:\n",
    "            #    ax.set_ylim(0, 2)\n",
    "                #ax.set_xlim(5, 20)\n",
    "            \n",
    "            #ax.set_title(method)\n",
    "        axes[jcount, 0].legend()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  },
  "vscode": {
   "interpreter": {
    "hash": "b50c9e5c20b80a7fea53278e7b85976e5483a9191b272239eb7a6e566d7885cc"
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
