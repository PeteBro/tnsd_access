Getting Started
===============

Installation
------------

Clone the repo and install in editable mode:

.. code-block:: bash

   git clone git@github.com:PeteBro/tnsd_access.git
   cd tnsd_access
   pip install -e .

Usage
-----

Load trials for a subject
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from tnsd_access import TrialHandler

   loader = TrialHandler()

   # Look up trials for subject 6, filtering to conditions 5 and 2951
   trials = loader.lookup_trials(subject=6, condition=[5, 2951])

   # Load all matching trials into memory
   # Returns {'data': ndarray, 'metadata': DataFrame}
   result = loader.get_data(trials)

Average by condition (ERP-style)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from tnsd_access import TrialHandler

   loader = TrialHandler()

   # Look up all shared trials across subjects
   trials = loader.lookup_trials(shared=True)

   # Load data and average across trials within each condition
   # result['data'] shape: (n_conditions, n_channels, n_samples)
   # result['metadata'] has one row per condition
   result = loader.get_data(trials, average_by='condition')

Filter options
~~~~~~~~~~~~~~

Pass any of the following as keyword arguments to ``lookup_trials()`` to
select a subset of trials.  Each accepts a single value or a list.

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Filter
     - Type
     - Description
   * - ``subject``
     - int
     - Subject number. Valid values: 1–18.
   * - ``condition``
     - int
     - NSD stimulus ID. Valid values: 1–73000.
   * - ``session``
     - int
     - Session number.
   * - ``run``
     - int
     - Run number within a session.
   * - ``epoch``
     - int
     - Epoch number within a run.
   * - ``trial_instance``
     - int
     - Repetition index for a trial_type + condition combination (e.g. ``1`` for
       first-presentation).
   * - ``trial_type``
     - str
     - Trial type label. Valid values: 'stimulus', 'recall', 'one-back'.
   * - ``shared``
     - bool
     - ``True`` to keep only the 1000 stimuli shared across all subjects;
       ``False`` for subject-unique stimuli.
   * - ``datetime``
     - str
     - UNIX time of trial onset in seconds.
   * - ``onset``
     - float
     - Trial onset time in seconds.
   * - ``duration``
     - float
     - Trial duration in seconds.
   * - ``response_time``
     - float
     - Participant response time in seconds.
   * - ``response``
     - str
     - Participant response value.
   * - ``trigger_value``
     - int
     - Trigger value sent at trial onset.
   * - ``event_id``
     - str
     - Unique SHA256 trial id.

Batch iteration (memory-efficient)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from tnsd_access import TrialHandler

   loader = TrialHandler()

   # Look up trials for subject 6
   trials = loader.lookup_trials(subject=6)

   # Iterate in batches of 32 trials — useful for large datasets
   for batch in loader.iter_data(trials, batch_size=32):
       data = batch['data']      # (batch_size, n_channels, n_samples)
       meta = batch['metadata']  # DataFrame aligned to data's first axis
