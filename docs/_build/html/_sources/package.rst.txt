Main Classes
============

.. contents::

Overview
~~~~~~~~

All classes are defined in submodules but directly imported in the main package, which means they are directly accessible in the pyfiber namespace.

.. code-block:: python

   import pyfiber as pf
   
   b = pf.Behavior('<filepath>')
   f = pf.Fiber('<filepath>')
   mb = pf.MultiBehavior('<folder>')
   s = pf.Session('<folder>')
   ms = pf.MultiSession('<folder>')

.. note::
   
   Analysis and MultiAnalysis instances are not directly created by users but returned by Session and MultiSession instances when analyzing peri-events.


Four main categories of objects exist:

**Behavior** and **MultiBehavior**
   As the name suggest, these classes extract and describe single or multiple behavioral data files. Currently, Imetronic '.dat' files are the default, but *csv* files can also be inputed (In that case they, should follow a specific structure, see :doc:`configuration`). 

**Fiber** 
   This class is used to extract and analyze fiber photometry data. Currently it accepts by default Doric systems *csv* files. Other file formats may be extracted, by providing the columns labels in the configuration file (see :doc:`configuration`)

**Session** and **MultiSession**
   Theses classes read data from both behavioral data files and fiber photometry files. Their main purpose is to allow precise peri-event analysis, around one or many defined types of events, respectively for one session or many.
   The analysis result is returned as an Analysis or MultiAnalysis instance, for convenience.

**Analysis** and **MultiAnalysis**
   Analysis result outputted by Session or MultiSession objects. They contain data, stored in attributes (for example, pre-event and post event standard scores or robust standard scores), optionnally outputted as dataframe. Finally, they can plot the traces around the event of interest.


Behavior and MultiBehavior
~~~~~~~~~~~~~~~~~~~~~~~~~~

Behavior
--------
.. autoclass:: pyfiber.Behavior
   :members:


MultiBehavior
-------------
.. autoclass:: pyfiber.MultiBehavior
   :members:
   :member-order: bysource
   :undoc-members:
   :private-members:


Fiber
~~~~~~
.. autoclass:: pyfiber.Fiber
   :members:
   :member-order: bysource
   :undoc-members:
   :private-members:


Session and MultiSession
~~~~~~~~~~~~~~~~~~~~~~~~

Session
-------

.. autoclass:: pyfiber.Session
   :members:
   :member-order: bysource
   :undoc-members:
   :private-members:

MultiSession
------------

.. autoclass:: pyfiber.MultiSession
   :members:
   :member-order: bysource
   :undoc-members:
   :private-members:

Analysis and MultiAnalysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

Analysis
--------
.. autoclass:: pyfiber.Analysis
   :members:
   :member-order: bysource
   :undoc-members:
   :private-members:

MultiAnalysis
-------------
.. autoclass:: pyfiber.MultiAnalysis
   :members:
   :member-order: bysource
   :undoc-members:
   :private-members:






