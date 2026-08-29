(define (domain rais_orchestration)
  (:requirements :strips :typing :action-costs :negative-preconditions)

  (:types
    device
    stage
  )

  (:constants
    adversarial_training
    perturbation_generation
    metric_computation
    rais_scoring
    pruning_decision
    recovery_finetuning
    model_evaluation
    deploy_updated_model
    - stage
  )

  (:predicates
    ;; device state / capability
    (available ?d - device)
    (has-gpu ?d - device)
    (workstation-gpu ?d - device)
    (preferred-workstation ?d - device)
    (edge-device ?d - device)
    (high-mem ?d - device)
    (cpu-device ?d - device)
    (cpu-free ?d - device)
    (cpu-busy ?d - device)
    (low-latency ?d - device)
    (high-latency ?d - device)
    (low-mem ?d - device)
    (device-unreachable ?d - device)
    (device-failed ?d - device)

    ;; workflow state
    (ready ?s - stage)
    (completed ?s - stage)
    (assigned ?s - stage ?d - device)
  )

  (:functions
    (total-cost)
  )

  ;; --------------------------------------------------
  ;; 1) adversarial_training
  ;; --------------------------------------------------
  (:action assign-adversarial-training
    :parameters (?d - device)
    :precondition (and
      (ready adversarial_training)
      (available ?d)
      (has-gpu ?d)
      (not (device-failed ?d))
      (not (device-unreachable ?d))
    )
    :effect (assigned adversarial_training ?d)
  )

  (:action run-adversarial-training-workstation
    :parameters (?d - device)
    :precondition (and
      (assigned adversarial_training ?d)
      (ready adversarial_training)
      (available ?d)
      (has-gpu ?d)
      (workstation-gpu ?d)
    )
    :effect (and
      (completed adversarial_training)
      (not (ready adversarial_training))
      (ready perturbation_generation)
      (increase (total-cost) 3)
    )
  )

  (:action run-adversarial-training-edge
    :parameters (?d - device)
    :precondition (and
      (assigned adversarial_training ?d)
      (ready adversarial_training)
      (available ?d)
      (has-gpu ?d)
      (edge-device ?d)
    )
    :effect (and
      (completed adversarial_training)
      (not (ready adversarial_training))
      (ready perturbation_generation)
      (increase (total-cost) 8)
    )
  )

  ;; --------------------------------------------------
  ;; 2) perturbation_generation
  ;; --------------------------------------------------
  (:action assign-perturbation-generation
    :parameters (?d - device)
    :precondition (and
      (ready perturbation_generation)
      (available ?d)
      (not (device-failed ?d))
      (not (device-unreachable ?d))
    )
    :effect (assigned perturbation_generation ?d)
  )

  (:action run-perturbation-generation-gpu
    :parameters (?d - device)
    :precondition (and
      (assigned perturbation_generation ?d)
      (ready perturbation_generation)
      (available ?d)
      (has-gpu ?d)
    )
    :effect (and
      (completed perturbation_generation)
      (not (ready perturbation_generation))
      (ready metric_computation)
      (increase (total-cost) 2)
    )
  )

  (:action run-perturbation-generation-cpu
    :parameters (?d - device)
    :precondition (and
      (assigned perturbation_generation ?d)
      (ready perturbation_generation)
      (available ?d)
      (cpu-device ?d)
    )
    :effect (and
      (completed perturbation_generation)
      (not (ready perturbation_generation))
      (ready metric_computation)
      (increase (total-cost) 15)
    )
  )

  ;; --------------------------------------------------
  ;; 3) metric_computation
  ;; --------------------------------------------------
  (:action assign-metric-computation
    :parameters (?d - device)
    :precondition (and
      (ready metric_computation)
      (available ?d)
      (not (device-failed ?d))
      (not (device-unreachable ?d))
    )
    :effect (assigned metric_computation ?d)
  )

  (:action run-metric-computation-gpu
    :parameters (?d - device)
    :precondition (and
      (assigned metric_computation ?d)
      (ready metric_computation)
      (available ?d)
      (has-gpu ?d)
    )
    :effect (and
      (completed metric_computation)
      (not (ready metric_computation))
      (ready rais_scoring)
      (increase (total-cost) 2)
    )
  )

  (:action run-metric-computation-cpu
    :parameters (?d - device)
    :precondition (and
      (assigned metric_computation ?d)
      (ready metric_computation)
      (available ?d)
      (cpu-device ?d)
    )
    :effect (and
      (completed metric_computation)
      (not (ready metric_computation))
      (ready rais_scoring)
      (increase (total-cost) 12)
    )
  )

  ;; --------------------------------------------------
  ;; 4) rais_scoring
  ;; --------------------------------------------------
  (:action assign-rais-scoring
    :parameters (?d - device)
    :precondition (and
      (ready rais_scoring)
      (available ?d)
      (not (device-failed ?d))
      (not (device-unreachable ?d))
    )
    :effect (assigned rais_scoring ?d)
  )

  (:action run-rais-scoring-gpu
    :parameters (?d - device)
    :precondition (and
      (assigned rais_scoring ?d)
      (ready rais_scoring)
      (available ?d)
      (has-gpu ?d)
    )
    :effect (and
      (completed rais_scoring)
      (not (ready rais_scoring))
      (ready pruning_decision)
      (increase (total-cost) 2)
    )
  )

  (:action run-rais-scoring-cpu
    :parameters (?d - device)
    :precondition (and
      (assigned rais_scoring ?d)
      (ready rais_scoring)
      (available ?d)
      (cpu-device ?d)
    )
    :effect (and
      (completed rais_scoring)
      (not (ready rais_scoring))
      (ready pruning_decision)
      (increase (total-cost) 10)
    )
  )

  ;; --------------------------------------------------
  ;; 5) pruning_decision
  ;; --------------------------------------------------
  (:action assign-pruning-decision
    :parameters (?d - device)
    :precondition (and
      (ready pruning_decision)
      (available ?d)
      (not (device-failed ?d))
      (not (device-unreachable ?d))
    )
    :effect (assigned pruning_decision ?d)
  )

  (:action run-pruning-decision-gpu
    :parameters (?d - device)
    :precondition (and
      (assigned pruning_decision ?d)
      (ready pruning_decision)
      (available ?d)
      (has-gpu ?d)
    )
    :effect (and
      (completed pruning_decision)
      (not (ready pruning_decision))
      (ready recovery_finetuning)
      (increase (total-cost) 2)
    )
  )

  (:action run-pruning-decision-cpu
    :parameters (?d - device)
    :precondition (and
      (assigned pruning_decision ?d)
      (ready pruning_decision)
      (available ?d)
      (cpu-device ?d)
    )
    :effect (and
      (completed pruning_decision)
      (not (ready pruning_decision))
      (ready recovery_finetuning)
      (increase (total-cost) 8)
    )
  )

  ;; --------------------------------------------------
  ;; 6) recovery_finetuning
  ;; --------------------------------------------------
  (:action assign-recovery-finetuning
    :parameters (?d - device)
    :precondition (and
      (ready recovery_finetuning)
      (available ?d)
      (has-gpu ?d)
      (not (device-failed ?d))
      (not (device-unreachable ?d))
    )
    :effect (assigned recovery_finetuning ?d)
  )

  (:action run-recovery-finetuning-workstation
    :parameters (?d - device)
    :precondition (and
      (assigned recovery_finetuning ?d)
      (ready recovery_finetuning)
      (available ?d)
      (has-gpu ?d)
      (workstation-gpu ?d)
    )
    :effect (and
      (completed recovery_finetuning)
      (not (ready recovery_finetuning))
      (ready model_evaluation)
      (increase (total-cost) 3)
    )
  )

  (:action run-recovery-finetuning-edge
    :parameters (?d - device)
    :precondition (and
      (assigned recovery_finetuning ?d)
      (ready recovery_finetuning)
      (available ?d)
      (has-gpu ?d)
      (edge-device ?d)
    )
    :effect (and
      (completed recovery_finetuning)
      (not (ready recovery_finetuning))
      (ready model_evaluation)
      (increase (total-cost) 8)
    )
  )

  ;; --------------------------------------------------
  ;; 7) model_evaluation
  ;; --------------------------------------------------
  (:action assign-model-evaluation
    :parameters (?d - device)
    :precondition (and
      (ready model_evaluation)
      (available ?d)
      (not (device-failed ?d))
      (not (device-unreachable ?d))
    )
    :effect (assigned model_evaluation ?d)
  )

  (:action run-model-evaluation-gpu
    :parameters (?d - device)
    :precondition (and
      (assigned model_evaluation ?d)
      (ready model_evaluation)
      (available ?d)
      (has-gpu ?d)
    )
    :effect (and
      (completed model_evaluation)
      (not (ready model_evaluation))
      (ready deploy_updated_model)
      (increase (total-cost) 2)
    )
  )

  (:action run-model-evaluation-cpu
    :parameters (?d - device)
    :precondition (and
      (assigned model_evaluation ?d)
      (ready model_evaluation)
      (available ?d)
      (cpu-device ?d)
    )
    :effect (and
      (completed model_evaluation)
      (not (ready model_evaluation))
      (ready deploy_updated_model)
      (increase (total-cost) 10)
    )
  )

  ;; --------------------------------------------------
  ;; 8) deploy_updated_model
  ;; --------------------------------------------------
  (:action assign-deploy-updated-model
    :parameters (?d - device)
    :precondition (and
      (ready deploy_updated_model)
      (available ?d)
      (edge-device ?d)
      (cpu-device ?d)
      (not (device-failed ?d))
      (not (device-unreachable ?d))
    )
    :effect (assigned deploy_updated_model ?d)
  )

  (:action run-deploy-updated-model-cpu
    :parameters (?d - device)
    :precondition (and
      (assigned deploy_updated_model ?d)
      (ready deploy_updated_model)
      (available ?d)
      (edge-device ?d)
      (cpu-device ?d)
    )
    :effect (and
      (completed deploy_updated_model)
      (not (ready deploy_updated_model))
      (increase (total-cost) 1)
    )
  )
)